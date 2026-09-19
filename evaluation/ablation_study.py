import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.feature_store import DATASET_CONFIGS
from Graph_models.contextual_gate import PerFlowContextualMetaGating
from baselines_ml.metrics import calc_metrics_numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ABLATION_CONFIGS = ['full_3branch', 'no_ml_branch', 'no_global_branch', 'static_average']


def evaluate_ablation_run(dataset_name, run_id=0):
    ds_key = dataset_name.lower()
    cfg = DATASET_CONFIGS[ds_key]
    seq_len = cfg['seq_len']

    cache_file = os.path.join(parent_dir, 'cache', f"{ds_key}_run_{run_id}_test_preds.pt")
    gate_file = os.path.join(parent_dir, 'logs', f"st_adaptive_ensemble_data_{ds_key}_seq_{seq_len}", f"run_{run_id}", 'best_gate.pth')

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Không tìm thấy file cache test: {cache_file}")

    with open(cache_file, 'rb') as f:
        cached = torch.load(f, map_location=device)
    y_g = cached['y_global'].cpu().numpy()
    y_l = cached['y_local'].cpu().numpy()
    y_m = cached['y_ml'].cpu().numpy()
    y_real = cached['y_real'].cpu().numpy()
    ctx = cached['context']

    records = []

    # 1. full_3branch: Meta-Gating động 3 nhánh
    if os.path.exists(gate_file):
        gate = PerFlowContextualMetaGating(context_dim=ctx.shape[-1], hidden_dim=32)
        with open(gate_file, 'rb') as f:
            gate.load_state_dict(torch.load(f, map_location=device))
        gate.to(device)
        gate.eval()
        with torch.no_grad():
            w = gate(ctx.to(device)).cpu().numpy()
        y_full = w[:, :, 0] * y_g + w[:, :, 1] * y_l + w[:, :, 2] * y_m
    else:
        # Fallback trung bình nếu chưa train gate
        y_full = (y_g + y_l + y_m) / 3.0

    m_full = calc_metrics_numpy(y_full, y_real)
    m_full.update({'config': 'full_3branch', 'dataset': ds_key.upper(), 'run': run_id})
    records.append(m_full)

    # 2. no_ml_branch: Chỉ Global + Local (loại bỏ hoàn toàn nhánh ML)
    # Tỷ lệ chuẩn hóa giữa 2 nhánh DL
    y_no_ml = 0.5 * y_g + 0.5 * y_l
    m_no_ml = calc_metrics_numpy(y_no_ml, y_real)
    m_no_ml.update({'config': 'no_ml_branch', 'dataset': ds_key.upper(), 'run': run_id})
    records.append(m_no_ml)

    # 3. no_global_branch: Chỉ Local + ML (loại bỏ hoàn toàn Attention/Transformer)
    y_no_glob = 0.5 * y_l + 0.5 * y_m
    m_no_glob = calc_metrics_numpy(y_no_glob, y_real)
    m_no_glob.update({'config': 'no_global_branch', 'dataset': ds_key.upper(), 'run': run_id})
    records.append(m_no_glob)

    # 4. static_average: Gộp trung bình tĩnh 1/3 (w_1 = w_2 = w_3 = 1/3)
    y_static = (y_g + y_l + y_m) / 3.0
    m_static = calc_metrics_numpy(y_static, y_real)
    m_static.update({'config': 'static_average', 'dataset': ds_key.upper(), 'run': run_id})
    records.append(m_static)

    return records


def run_ablation_experiments(datasets=None, runs=5):
    if datasets is None or 'all' in datasets:
        datasets = ['sdn', 'geant', 'abilene']

    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(" THỰC HIỆN ABLATION STUDY (4 CẤU HÌNH ĐỐI CHỨNG)")
    print(f" Datasets: {datasets} | Runs: {runs}")
    print("=" * 80)

    all_records = []
    for ds in datasets:
        for r in range(runs):
            try:
                rec = evaluate_ablation_run(ds, run_id=r)
                all_records.extend(rec)
            except Exception as e:
                print(f"  [WARN] Ablation {ds.upper()} run {r}: {e}")

    df_ablation = pd.DataFrame(all_records)
    if not df_ablation.empty:
        out_csv = os.path.join(results_dir, 'ablation_results.csv')
        df_ablation.to_csv(out_csv, index=False)

        # Tổng hợp Mean ± Std cho từng cấu hình
        summary = df_ablation.groupby(['dataset', 'config']).agg(
            mean_mse=('mse', lambda x: np.mean(x) * 1000.0),
            std_mse=('mse', lambda x: np.std(x) * 1000.0),
            mean_mae=('mae', lambda x: np.mean(x) * 1000.0),
            mean_rmse=('rmse', 'mean')
        ).reset_index()

        summary_csv = os.path.join(results_dir, 'ablation_summary.csv')
        summary.to_csv(summary_csv, index=False)

        print("\nBẢNG TỔNG HỢP ABLATION STUDY (MSE x 10^-3):")
        print(summary.to_string(index=False))
        print(f"\n-> Đã lưu bảng kết quả Ablation tại: {out_csv}")

    return df_ablation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    parser.add_argument('--datasets', type=str, default='all', help="Comma-separated datasets hoặc 'all'")
    parser.add_argument('--runs', type=int, default=5, help="Số run")

    args = parser.parse_args()
    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ['sdn', 'geant', 'abilene']
    run_ablation_experiments(datasets=d_list, runs=args.runs)
