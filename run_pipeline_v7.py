"""
Điều phối pipeline v7.

    # Giai đoạn 0 - xác thực trước, ~10 phút, KHÔNG cần GPU
    python tools/quick_validate_v7.py

    # Giai đoạn 1 - làm sạch (BẮT BUỘC: cache/logs hiện tại trộn hai pipeline)
    python run_pipeline_v7.py --stage clean
    python run_pipeline_v7.py --stage baselines --dataset all

    # Giai đoạn 2 - Stage A: huấn luyện nhánh (một dataset mỗi session Kaggle)
    python run_pipeline_v7.py --stage branches --dataset sdn     --runs 10
    python run_pipeline_v7.py --stage branches --dataset geant   --runs 10
    python run_pipeline_v7.py --stage branches --dataset abilene --runs 10

    # Giai đoạn 3 - cache + bộ gộp + ablation (CPU, nhanh)
    python run_pipeline_v7.py --stage combine --dataset all --runs 10

    # Giai đoạn 4 - Stage B: OOF rồi chạy lại combine
    python run_pipeline_v7.py --stage oof --dataset all --runs 10 --folds 5
"""
import os
import sys
import shutil
import argparse
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASETS = ['sdn', 'geant', 'abilene']


def stage_clean(_):
    """Xoá cache/, logs/, results/ để chạy lại trên MỘT phiên bản pipeline duy nhất.

    Bắt buộc, không phải tuỳ chọn: hiện có hai phiên bản trộn lẫn (run 0-4 của
    Géant/Abilene sinh từ pipeline cũ). Mọi std trong báo cáo hiện tại là artifact.
    Thư mục cũ được ĐỔI TÊN chứ không xoá hẳn, để còn đối chiếu khi viết luận văn.
    """
    for d in ['cache', 'logs', 'results']:
        if os.path.isdir(d):
            bak = d + '_v6_backup'
            if os.path.exists(bak):
                shutil.rmtree(bak)
            shutil.move(d, bak)
            print(f"  Đã chuyển {d}/ -> {bak}/ (giữ để đối chiếu, không dùng trong v7)")
        os.makedirs(d, exist_ok=True)


def stage_baselines(args):
    """G1 - baseline tầm thường, chạy TRƯỚC khi huấn luyện bất cứ mô hình nào."""
    from features.feature_store import prepare_feature_store, DATASET_CONFIGS
    from evaluation.baselines_naive import historical_average

    rows = []
    for ds in args.datasets:
        *_, meta = prepare_feature_store(ds)
        seq_len = DATASET_CONFIGS[ds]['seq_len']
        te = np.asarray(meta['test_norm'], dtype=np.float64)
        y = te[seq_len:]

        rows.append({'dataset': ds, 'model': 'persistence',
                     'mse': float(((te[seq_len - 1:-1] - y) ** 2).mean())})

        # Seasonal naive: y_hat[t] = x[t - period]. Chỉ tính trên các cửa sổ mà mốc
        # t-period còn nằm trong tập Test; các cửa sổ đầu bị bỏ chứ không đệm, vì
        # đệm sẽ làm baseline đẹp lên một cách giả tạo (I2 - không cắt/đệm ngầm).
        period = 288
        i0 = max(0, period - seq_len)
        if len(y) > i0:
            src = te[seq_len + i0 - period: seq_len + len(y) - period]
            rows.append({'dataset': ds, 'model': 'seasonal_naive',
                         'mse': float(((src - y[i0:]) ** 2).mean()),
                         'n_eval': len(src)})

        ha = historical_average(meta['train_norm'], meta['tod_train'], meta['dow_train'],
                                meta['tod_test'][seq_len:], meta['dow_test'][seq_len:])
        rows.append({'dataset': ds, 'model': 'historical_average',
                     'mse': float(((ha - y) ** 2).mean())})

    df = pd.DataFrame(rows)
    df['mse_x1e3'] = df['mse'] * 1e3
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/baselines_naive.csv', index=False)
    print(df.to_string(index=False))
    print("\n  -> results/baselines_naive.csv")
    print("  Bắt buộc đưa vào luận văn: persistence tốt hơn GWN đã công bố trên Géant")
    print("  (0.855 vs 0.879) và tốt hơn 62% trên Abilene (2.363 vs 6.220).")


def stage_branches(args):
    """G2 - huấn luyện ba nhánh. Uỷ quyền cho run_experiments.py."""
    import subprocess
    for ds in args.datasets:
        for model in ['STWaveFormer', 'LocalSpatialTCN']:
            cmd = [sys.executable, 'run_experiments.py', '--dataset', ds,
                   '--model', model, '--runs', str(args.runs)]
            print(f"  $ {' '.join(cmd)}", flush=True)
            if subprocess.call(cmd) != 0:
                raise RuntimeError(f"Huấn luyện {model}/{ds} thất bại.")
        cmd = [sys.executable, '-m', 'baselines_ml.run_ml_baselines',
               '--dataset', ds, '--runs', str(args.runs)]
        print(f"  $ {' '.join(cmd)}", flush=True)
        subprocess.call(cmd)


def _load(ds, run, split):
    path = os.path.join('cache', f'{ds}_run_{run}_{split}_preds.pt')
    d = torch.load(path, weights_only=False)
    P = np.stack([d['y_global'].numpy(), d['y_local'].numpy(), d['y_ml'].numpy()],
                 -1).astype(np.float64)
    x = d['x_last'].numpy().astype(np.float64) if 'x_last' in d else None
    return P, x, d['context'].numpy().astype(np.float32), d['y_real'].numpy().astype(np.float64)


def stage_combine(args):
    """G4-G6 - bộ gộp + ablation trên cache đã sinh."""
    from training.train_pfar import select_hyperparams
    from evaluation.ablation_v7 import run_ablation, summarize

    for ds in args.datasets:
        rows, hps = [], []
        for r in range(args.runs):
            if not os.path.exists(os.path.join('cache', f'{ds}_run_{r}_test_preds.pt')):
                continue
            Pv, xv, cv, yv = _load(ds, r, 'val')
            Pt, xt, ct, yt = _load(ds, r, 'test')
            if xv is None or xt is None:
                raise RuntimeError(
                    f"Cache {ds} run {r} không có 'x_last'. "
                    f"Hãy chạy lại training/precompute_cache.py của v7.")

            oof_path = os.path.join('cache', f'{ds}_run_{r}_oof.pt')
            P_oof = x_oof = c_oof = y_oof = None
            if os.path.exists(oof_path):
                o = torch.load(oof_path, weights_only=False)
                P_oof = o['P_oof'].numpy().astype(np.float64)
                y_oof = o['y_oof'].numpy().astype(np.float64)
                x_oof = o['x_last'].numpy().astype(np.float64)
                c_oof = np.zeros(P_oof.shape[:2] + (cv.shape[-1],), dtype=np.float32)

            # Khối chọn siêu tham số: OOF nếu có, nếu không thì Val. KHÔNG BAO GIỜ Test (I1).
            sel_P, sel_y = (P_oof, y_oof) if P_oof is not None else (Pv, yv)
            hp = select_hyperparams(sel_P, sel_y, mode='online', verbose=True)
            hps.append(hp)
            rows.append(run_ablation(Pv, xv, cv, yv, Pt, xt, ct, yt, hp,
                                     P_oof, x_oof, c_oof, y_oof))

        if not rows:
            print(f"  [{ds}] chưa có cache, bỏ qua.")
            continue

        summarize(rows, ds)
        df = pd.DataFrame(rows)
        for k in ('ridge', 'lam', 'anchor'):
            df[k] = [h[k] for h in hps]
        df.insert(0, 'run', range(len(df)))
        df.insert(0, 'dataset', ds)
        os.makedirs('results', exist_ok=True)
        df.to_csv(f'results/ablation_v7_{ds}.csv', index=False)
        print(f"  -> results/ablation_v7_{ds}.csv")


def stage_oof(args):
    from training.build_oof import build_oof_matrix
    for ds in args.datasets:
        for r in range(args.runs):
            build_oof_matrix(ds, r, K=args.folds)


STAGES = {
    'clean': stage_clean,
    'baselines': stage_baselines,
    'branches': stage_branches,
    'combine': stage_combine,
    'oof': stage_oof,
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=list(STAGES))
    ap.add_argument('--dataset', default='all')
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()
    a.datasets = DATASETS if a.dataset == 'all' else [x.strip() for x in a.dataset.split(',')]
    STAGES[a.stage](a)
