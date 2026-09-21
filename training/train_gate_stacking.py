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
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.feature_store import DATASET_CONFIGS
from Graph_models.contextual_gate import PerFlowContextualMetaGating
from baselines_ml.metrics import calc_metrics_numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hàm mất mát của cổng. 'huber' (delta=0.01) là thiết kế NGUYÊN BẢN của
# ST-Adaptive-Ensemble và được giữ làm mặc định. 'mse' là biến thể để so sánh
# công bằng với PFAR: trên thang MinMax [0,1] sai số điển hình nhỏ hơn 0.01 nên
# Huber(delta=0.01) gần như luôn ở vùng tuyến tính, tức thực chất tối ưu MAE trong
# khi chỉ số báo cáo là MSE (lỗi B3, xem acceptance_gates.c4_loss_matches_metric).
GATE_LOSSES = ('huber', 'mse')
# Hậu tố tên thư mục / tệp kết quả, để hai biến thể không ghi đè lên nhau.
_LOSS_SUFFIX = {'huber': ('', ''), 'mse': ('_mse', 'MSE')}


def train_gate_for_run(dataset_name, run_id=0, epochs=100, lr=1e-3, lambda_entropy=0.001, logdir=None,
                       loss='huber'):
    if loss not in GATE_LOSSES:
        raise ValueError(f"loss không hợp lệ: {loss!r}. Chọn một trong {GATE_LOSSES}.")
    ds_key = dataset_name.lower()
    cfg = DATASET_CONFIGS[ds_key]
    seq_len = cfg['seq_len']

    cache_dir = os.path.join(parent_dir, 'cache')
    val_cache_file = os.path.join(cache_dir, f"{ds_key}_run_{run_id}_val_preds.pt")
    test_cache_file = os.path.join(cache_dir, f"{ds_key}_run_{run_id}_test_preds.pt")

    if not os.path.exists(val_cache_file) or not os.path.exists(test_cache_file):
        raise FileNotFoundError(
            f"Không tìm thấy file cache cho {ds_key} run {run_id}. "
            f"Vui lòng chạy: python training/precompute_cache.py --datasets {ds_key} --runs {run_id+1} trước!"
        )

    with open(val_cache_file, 'rb') as f:
        cached_val = torch.load(f, map_location=device)
    with open(test_cache_file, 'rb') as f:
        cached_test = torch.load(f, map_location=device)

    # 1. Tính Empirical Loss Prior từ tập Validation
    y_real_val = cached_val['y_real']
    mse_glob = F.mse_loss(cached_val['y_global'], y_real_val).item()
    mse_loc = F.mse_loss(cached_val['y_local'], y_real_val).item()
    mse_ml = F.mse_loss(cached_val['y_ml'], y_real_val).item()

    eps = 1e-6
    prior_logits = torch.tensor([
        -np.log(mse_glob + eps),
        -np.log(mse_loc + eps),
        -np.log(mse_ml + eps)
    ], dtype=torch.float32, device=device)

    print(f"      [Prior] Val MSE: Global={mse_glob*1000:.3f}e-3, Local={mse_loc*1000:.3f}e-3, ML={mse_ml*1000:.3f}e-3")
    print(f"      [Prior] Initial Logits: {prior_logits.cpu().numpy().round(3)}")

    # 2. Khởi tạo mạng Gate
    context_dim = cached_val['context'].shape[-1]
    gate = PerFlowContextualMetaGating(context_dim=context_dim, hidden_dim=32, init_prior=prior_logits)
    gate.to(device)

    optimizer = optim.Adam(gate.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 3. Huấn luyện Gate trên tập Validation (Offline Cache Stacking)
    t0 = time.perf_counter()
    best_loss = float('inf')
    best_weights = None
    best_state = None

    ctx_val = cached_val['context']
    y_g_val = cached_val['y_global']
    y_l_val = cached_val['y_local']
    y_m_val = cached_val['y_ml']

    for epoch in range(epochs):
        gate.train()
        weights = gate(ctx_val) # [M, N, 3]

        y_hat = (weights[:, :, 0] * y_g_val +
                 weights[:, :, 1] * y_l_val +
                 weights[:, :, 2] * y_m_val)

        if loss == 'mse':
            loss_fit = F.mse_loss(y_hat, y_real_val)
        else:
            loss_fit = F.huber_loss(y_hat, y_real_val, delta=0.01)
        # Entropy regularization khuyến khích trọng số dứt khoát
        entropy = -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=-1))
        loss_total = loss_fit + lambda_entropy * entropy

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            best_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}

    train_time_s = time.perf_counter() - t0
    gate.load_state_dict(best_state)
    gate.eval()

    # 4. Đánh giá trên tập Test
    ctx_test = cached_test['context']
    y_g_test = cached_test['y_global']
    y_l_test = cached_test['y_local']
    y_m_test = cached_test['y_ml']
    y_real_test = cached_test['y_real']

    t_inf_0 = time.perf_counter()
    with torch.no_grad():
        w_test = gate(ctx_test) # [M, N, 3]
        y_hat_test = (w_test[:, :, 0] * y_g_test +
                      w_test[:, :, 1] * y_l_test +
                      w_test[:, :, 2] * y_m_test)
    t_inf_1 = time.perf_counter()

    num_test_batches = int(np.ceil(len(ctx_test) / 64))
    inf_time_ms = ((t_inf_1 - t_inf_0) * 1000.0) / max(1, num_test_batches)

    y_pred_np = y_hat_test.cpu().numpy()
    y_real_np = y_real_test.cpu().numpy()
    w_test_np = w_test.cpu().numpy()

    metrics = calc_metrics_numpy(y_pred_np, y_real_np)
    metrics['inference_time_ms'] = float(inf_time_ms)
    metrics['gate_train_time_s'] = float(train_time_s)
    metrics['mean_w_global'] = float(np.mean(w_test_np[:, :, 0]))
    metrics['mean_w_local'] = float(np.mean(w_test_np[:, :, 1]))
    metrics['mean_w_ml'] = float(np.mean(w_test_np[:, :, 2]))
    metrics['gate_loss'] = loss

    # Lưu trữ checkpoint và logs
    if logdir is None:
        dir_sfx = _LOSS_SUFFIX[loss][0]
        logdir = os.path.join(parent_dir, 'logs', f"st_adaptive_ensemble{dir_sfx}_data_{ds_key}_seq_{seq_len}",
                              f"run_{run_id}")
    os.makedirs(logdir, exist_ok=True)

    with open(os.path.join(logdir, 'best_gate.pth'), 'wb') as f:
        torch.save(best_state, f)
    pd.DataFrame([metrics]).to_csv(os.path.join(logdir, 'test_metrics.csv'), index=False)
    np.save(os.path.join(logdir, 'y_pred_data.npy'), y_pred_np)
    np.save(os.path.join(logdir, 'y_real_data.npy'), y_real_np)
    np.save(os.path.join(logdir, 'gate_weights.npy'), w_test_np)

    print(f"      [Test] MSE: {metrics['mse']*1000.0:.3f}e-3 | MAE: {metrics['mae']*1000.0:.3f}e-3 | Train Time: {train_time_s:.2f}s")
    print(f"      [Gate Weights] Global: {metrics['mean_w_global']*100:.1f}% | Local: {metrics['mean_w_local']*100:.1f}% | ML: {metrics['mean_w_ml']*100:.1f}%")

    return metrics


def train_all_gates(datasets=None, runs=5, epochs=100, lr=1e-3, loss='huber'):
    if datasets is None or 'all' in datasets:
        datasets = ['sdn', 'geant', 'abilene']

    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(" HUẤN LUYỆN CỔNG ĐIỀU PHỐI ĐỘNG (TWO-STAGE STACKING GATE)")
    print(f" Datasets: {datasets} | Runs: {runs} | Epochs: {epochs} | LR: {lr} | Loss: {loss}")
    print("=" * 80)

    for ds in datasets:
        print(f"\n---> Dataset: {ds.upper()}...")
        run_metrics = []
        for r in range(runs):
            print(f"  [*] Huấn luyện Gate [Run {r+1}/{runs}]...")
            m = train_gate_for_run(ds, run_id=r, epochs=epochs, lr=lr, loss=loss)
            m['run'] = r
            run_metrics.append(m)

        df_runs = pd.DataFrame(run_metrics)
        out_csv = os.path.join(results_dir,
                               f"results_STAdaptiveEnsemble{_LOSS_SUFFIX[loss][1]}_data_{ds.lower()}.csv")
        df_runs.to_csv(out_csv, index=False)

        mean_mse = df_runs['mse'].mean()
        mean_mae = df_runs['mae'].mean()
        print(f"\n[*] KẾT QUẢ TRUNG BÌNH ST-ADAPTIVE-ENSEMBLE ({ds.upper()}):")
        print(f"    MSE: {mean_mse*1000.0:.3f}e-3 | MAE: {mean_mae*1000.0:.3f}e-3 | W_Global: {df_runs['mean_w_global'].mean()*100:.1f}% | W_ML+Local: {(df_runs['mean_w_local'].mean()+df_runs['mean_w_ml'].mean())*100:.1f}%\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Contextual Meta-Gating with Stacking (Stage 2)")
    parser.add_argument('--datasets', type=str, default='all', help="Comma-separated datasets hoặc 'all'")
    parser.add_argument('--runs', type=int, default=5, help="Số run độc lập (mặc định 5)")
    parser.add_argument('--epochs', type=int, default=100, help="Số epoch huấn luyện Gate (5-10s)")
    parser.add_argument('--lr', type=float, default=0.001, help="Tốc độ học")
    parser.add_argument('--loss', type=str, default='huber', choices=list(GATE_LOSSES),
                        help="huber = thiết kế nguyên bản (mặc định); mse = biến thể so sánh công bằng với PFAR")

    args = parser.parse_args()
    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ['sdn', 'geant', 'abilene']
    train_all_gates(datasets=d_list, runs=args.runs, epochs=args.epochs, lr=args.lr, loss=args.loss)
