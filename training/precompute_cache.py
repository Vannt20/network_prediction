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
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.feature_store import prepare_feature_store, DATASET_CONFIGS
from features.manifest import load_manifest, assert_compatible, save_manifest
from features.spatial_features import build_physical_flow_adjacency
from features.temporal_features import extract_context_features_torch
from Graph_models.st_waveformer import STWaveFormer
from Graph_models.local_filters import SpatialDilatedTCN, load_local_branch
from baselines_ml.lgbm_baseline import LGBMBaseline
from baselines_ml.xgboost_baseline import XGBoostBaseline
from baselines_ml.catboost_baseline import CatBoostBaseline
from baselines_ml.tree_baselines import ExtraTreesBaseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_sliding_windows(traffic_norm, tod_arr, dow_arr, seq_len):
    """
    Tạo tensor cửa sổ trượt [M, seq_len, N, 3] và nhãn y [M, N]
    Khớp 100% với cách xử lý của TrafficDataset trong run_experiments.py.
    """
    N = traffic_norm.shape[1]
    if tod_arr.ndim == 1:
        tod_2d = np.tile(tod_arr[:, None], (1, N))
    else:
        tod_2d = tod_arr
    if dow_arr.ndim == 1:
        dow_2d = np.tile(dow_arr[:, None], (1, N))
    else:
        dow_2d = dow_arr

    comb = np.stack([traffic_norm, tod_2d, dow_2d], axis=-1).astype(np.float32)
    
    xs, ys = [], []
    for i in range(len(comb) - seq_len):
        xs.append(comb[i : i + seq_len])
        ys.append(comb[i + seq_len, :, 0])
        
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)


def _fit_args(X_tr, y_tr, es_frac=0.15):
    """Cắt 15% cuối của Train làm tập early stopping (Train-ES).

    I5 / lỗi B4: trước đây nhánh ML early-stop trên X_va_tab[:1000], tức 1000 dòng
    đầu của chính tập Validation mà bộ gộp dùng để khớp trọng số. Nhánh ML vì thế
    "đẹp giả" đúng ở nơi ra quyết định.
    """
    n = len(X_tr)
    cut = max(1, int(n * (1.0 - es_frac)))
    return X_tr[:cut], y_tr[:cut], X_tr[cut:], y_tr[cut:]


def compute_x_last(traffic_norm, seq_len):
    """Giá trị bước t-1 tương ứng với từng cửa sổ trượt -> baseline persistence.

    Cửa sổ i phủ [i, i+seq_len) và dự báo bước i+seq_len, nên bước liền trước là
    i+seq_len-1, tức hàng cuối của chính cửa sổ đó.
    """
    import numpy as _np
    M = len(traffic_norm) - seq_len
    return torch.tensor(_np.asarray(traffic_norm[seq_len - 1: seq_len - 1 + M], dtype=_np.float32))


def get_champion_name(results_dir):
    json_path = os.path.join(results_dir, 'champion_ml_model.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                return info.get('champion_model', 'lightgbm')
        except Exception:
            pass
    return 'lightgbm'


def precompute_dataset_cache(dataset_name, runs=5, champion_name=None, quick_check=False):
    ds_key = dataset_name.lower()
    cfg = DATASET_CONFIGS[ds_key]
    seq_len = cfg['seq_len']
    num_nodes = cfg['nodes']
    num_flows = cfg['flows']

    cache_dir = os.path.join(parent_dir, 'cache')
    logs_dir = os.path.join(parent_dir, 'logs')
    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(cache_dir, exist_ok=True)

    if champion_name is None:
        champion_name = get_champion_name(results_dir)

    print(f"\n=======================================================", flush=True)
    print(f"[*] PRECOMPUTE CACHE: {ds_key.upper()} (nodes={num_nodes}, flows={num_flows}, seq_len={seq_len})", flush=True)
    print(f"    Champion ML Model: {champion_name.upper()} | Runs: {runs} | Quick Check: {quick_check}", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Nạp và chuẩn bị dữ liệu
    (X_tr_tab, y_tr_tab), (X_va_tab, y_va_tab), (X_te_tab, y_te_tab), meta = prepare_feature_store(
        ds_key, align_to_seq_len=True, data_dir=os.path.join(parent_dir, 'data')
    )

    # 2. Tạo sliding windows cho Deep Learning
    x_val_win, y_val_win = create_sliding_windows(
        meta['val_norm'], meta['tod_val'], meta['dow_val'], seq_len
    )
    x_test_win, y_test_win = create_sliding_windows(
        meta['test_norm'], meta['tod_test'], meta['dow_test'], seq_len
    )

    if quick_check:
        limit_samples = 10
        x_val_win = x_val_win[:limit_samples]
        y_val_win = y_val_win[:limit_samples]
        x_test_win = x_test_win[:limit_samples]
        y_test_win = y_test_win[:limit_samples]
        X_va_tab = X_va_tab[:limit_samples * num_flows]
        y_va_tab = y_va_tab[:limit_samples * num_flows]
        X_te_tab = X_te_tab[:limit_samples * num_flows]
        y_te_tab = y_te_tab[:limit_samples * num_flows]

    # 3. Trích xuất Context Features cho Gate [M, N, 4]
    ctx_val = extract_context_features_torch(x_val_win)   # [M_val, N, 4]
    ctx_test = extract_context_features_torch(x_test_win) # [M_test, N, 4]

    # Baseline persistence, lưu sẵn vào cache để mọi bảng kết quả đều có nó (G1).
    x_last_val = compute_x_last(meta['val_norm'], seq_len)[:len(x_val_win)]
    x_last_test = compute_x_last(meta['test_norm'], seq_len)[:len(x_test_win)]
    if len(x_last_val) != len(x_val_win) or len(x_last_test) != len(x_test_win):
        raise RuntimeError("x_last lệch kích thước với cửa sổ trượt - không được cắt ngầm (I2).")

    for run_id in range(runs):
        print(f"  --> Xử lý Run {run_id + 1}/{runs}...", flush=True)

        # -------------------------------------------------------------
        # A. NHÁNH GLOBAL: ST-WaveFormer
        # -------------------------------------------------------------
        global_log_dir = os.path.join(logs_dir, f"stwaveformer_data_{ds_key}_seq_{seq_len}", f"run_{run_id}")
        ckpt_global = os.path.join(global_log_dir, 'best_model.pth')
        y_pred_global_test_file = os.path.join(global_log_dir, 'y_pred_data.npy')

        global_model = STWaveFormer(input_dim=num_flows, num_nodes=num_nodes, seq_len=seq_len, d_model=64, num_layers=2)
        # Không được im lặng chạy tiếp với trọng số khởi tạo ngẫu nhiên: cache sinh
        # ra khi đó trông hợp lệ nhưng vô nghĩa.
        if not os.path.exists(ckpt_global):
            raise FileNotFoundError(
                f"Không có checkpoint {ckpt_global}. Không được dùng trọng số khởi tạo ngẫu nhiên. "
                f"Hãy huấn luyện nhánh Global cho run {run_id} trước."
            )
        with open(ckpt_global, 'rb') as f:
            global_model.load_state_dict(torch.load(f, map_location=device))
        print(f"      [Global] Đã nạp checkpoint: {ckpt_global}", flush=True)

        # I3 - đối chiếu manifest của artifact với manifest của dữ liệu hiện tại.
        run_manifest = load_manifest(global_log_dir)
        assert_compatible(meta['manifest'], run_manifest)

        global_model.to(device)
        global_model.eval()

        # Suy diễn Validation cho Global
        val_loader = DataLoader(TensorDataset(x_val_win), batch_size=64, shuffle=False)
        preds_glob_val = []
        with torch.no_grad():
            for (bx,) in val_loader:
                out = global_model(bx.to(device))
                preds_glob_val.append(out.cpu())
        y_global_val = torch.cat(preds_glob_val, dim=0)

        # Suy diễn Test cho Global.
        # I2 - LUÔN suy diễn lại từ checkpoint, đối xứng với nhánh Local.
        # Trước đây khối này nạp lại y_pred_data.npy nếu file tồn tại và CẮT NGẦM
        # khi lệch kích thước. Đó chính là lỗi B2: run 0-4 của Abilene giữ artifact
        # 9596 dòng từ pipeline cũ, bị cắt xuống 9539 dòng nên lệch trục thời gian,
        # đẩy MSE nhánh Global từ 1.98e-3 lên 10.38e-3. Toàn bộ std 10 run trong
        # báo cáo cũ là artifact kỹ thuật, không phải phương sai do hạt giống.
        test_loader = DataLoader(TensorDataset(x_test_win), batch_size=64, shuffle=False)
        preds_glob_te = []
        with torch.no_grad():
            for (bx,) in test_loader:
                out = global_model(bx.to(device))
                preds_glob_te.append(out.cpu())
        y_global_test = torch.cat(preds_glob_te, dim=0)

        # -------------------------------------------------------------
        # B. NHÁNH LOCAL: SpatialDilatedTCN (LocalSpatialTCN)
        # -------------------------------------------------------------
        columns = meta['columns']
        adj_flow = build_physical_flow_adjacency(ds_key, columns)
        print(f"      [Topo] Đã nạp ma trận kề vật lý cấp độ luồng: shape={adj_flow.shape}", flush=True)

        local_model, local_src = load_local_branch(
            ds_key=ds_key,
            seq_len=seq_len,
            run_id=run_id,
            num_flows=num_flows,
            adj_flow=adj_flow,
            logs_dir=logs_dir,
            device=device
        )

        preds_loc_val = []
        with torch.no_grad():
            for (bx,) in val_loader:
                out = local_model(bx.to(device))
                preds_loc_val.append(out.cpu())
        y_local_val = torch.cat(preds_loc_val, dim=0)

        test_loader = DataLoader(TensorDataset(x_test_win), batch_size=64, shuffle=False)
        preds_loc_te = []
        with torch.no_grad():
            for (bx,) in test_loader:
                out = local_model(bx.to(device))
                preds_loc_te.append(out.cpu())
        y_local_test = torch.cat(preds_loc_te, dim=0)

        # -------------------------------------------------------------
        # C. NHÁNH ML: Champion GBDT Model
        # -------------------------------------------------------------
        ml_log_dir = os.path.join(logs_dir, f"{champion_name}_data_{ds_key}_shared", f"run_{run_id}")
        ml_model_file = os.path.join(ml_log_dir, 'model.bin')
        ml_pred_file = os.path.join(ml_log_dir, 'y_pred_data.npy')

        if champion_name == 'lightgbm':
            ml_inst = LGBMBaseline(n_estimators=20 if quick_check else 1000, random_state=42 + run_id)
        elif champion_name == 'xgboost':
            ml_inst = XGBoostBaseline(n_estimators=20 if quick_check else 1000, random_state=42 + run_id)
        elif champion_name == 'catboost':
            ml_inst = CatBoostBaseline(iterations=20 if quick_check else 1000, random_seed=42 + run_id)
        elif champion_name == 'extra_trees':
            ml_inst = ExtraTreesBaseline(n_estimators=10 if quick_check else 200, random_state=42 + run_id)
        else:
            # Mặc định an toàn: LightGBM (tốc độ cao nhất, tiết kiệm RAM)
            ml_inst = LGBMBaseline(n_estimators=20 if quick_check else 1000, random_state=42 + run_id)

        if os.path.exists(ml_model_file):
            try:
                ml_inst.load(ml_model_file)
                print(f"      [ML] Đã nạp mô hình Champion: {ml_model_file}", flush=True)
            except Exception as e:
                print(f"      [ML] Lỗi nạp model ({e}), huấn luyện lại...", flush=True)
                sample_tr = min(5000, len(X_tr_tab)) if quick_check else len(X_tr_tab)
                ml_inst.fit(*_fit_args(X_tr_tab[:sample_tr], y_tr_tab[:sample_tr]))
        else:
            print(f"      [ML] Huấn luyện Champion {champion_name.upper()}...", flush=True)
            sample_tr = min(5000, len(X_tr_tab)) if quick_check else len(X_tr_tab)
            ml_inst.fit(*_fit_args(X_tr_tab[:sample_tr], y_tr_tab[:sample_tr]))
            try:
                ml_inst.save(ml_model_file)
            except Exception:
                pass

        # Dự đoán ML trên Validation và Test
        val_ml_preds = ml_inst.predict(X_va_tab)
        test_ml_preds = ml_inst.predict(X_te_tab)

        y_ml_val = torch.tensor(val_ml_preds.reshape(len(x_val_win), num_flows), dtype=torch.float32)
        y_ml_test = torch.tensor(test_ml_preds.reshape(len(x_test_win), num_flows), dtype=torch.float32)

        # -------------------------------------------------------------
        # D. LƯU BỘ NHỚ ĐỆM (CACHE SAVING)
        # -------------------------------------------------------------
        cache_val_file = os.path.join(cache_dir, f"{ds_key}_run_{run_id}_val_preds.pt")
        cache_test_file = os.path.join(cache_dir, f"{ds_key}_run_{run_id}_test_preds.pt")

        from dataclasses import asdict
        mf = meta['manifest']
        mf_d = mf if isinstance(mf, dict) else asdict(mf)

        cache_val_data = {
            'y_global': y_global_val,
            'y_local': y_local_val,
            'y_ml': y_ml_val,
            'context': ctx_val,
            'y_real': y_val_win,
            'x_last': x_last_val,
            'manifest': mf_d,
        }

        cache_test_data = {
            'y_global': y_global_test,
            'y_local': y_local_test,
            'y_ml': y_ml_test,
            'context': ctx_test,
            'y_real': y_test_win,
            'x_last': x_last_test,
            'manifest': mf_d,
        }

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_val_file, 'wb') as f:
            torch.save(cache_val_data, f)
        with open(cache_test_file, 'wb') as f:
            torch.save(cache_test_data, f)
        print(f"      [CACHE] Đã lưu cache Validation: {cache_val_file}", flush=True)
        print(f"      [CACHE] Đã lưu cache Test: {cache_test_file}", flush=True)


def run_precompute(datasets=None, runs=5, champion=None, quick_check=False):
    if datasets is None or 'all' in datasets:
        datasets = ['sdn', 'geant', 'abilene']
    for ds in datasets:
        precompute_dataset_cache(ds, runs=runs, champion_name=champion, quick_check=quick_check)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Precompute predictions cache for Stage 2 Gate Training")
    parser.add_argument('--datasets', type=str, default='all', help="Comma-separated datasets: sdn,geant,abilene hoặc 'all'")
    parser.add_argument('--runs', type=int, default=5, help="Số run (mặc định 5)")
    parser.add_argument('--champion', type=str, default=None, help="Tên model champion ML (nếu None sẽ tự phát hiện)")
    parser.add_argument('--quick_check', action='store_true', help="Chế độ kiểm tra nhanh")

    args = parser.parse_args()
    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ['sdn', 'geant', 'abilene']
    run_precompute(datasets=d_list, runs=args.runs, champion=args.champion, quick_check=args.quick_check)
