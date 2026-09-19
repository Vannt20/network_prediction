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
import time
import argparse
import numpy as np
import pandas as pd

# sys.path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from features.feature_store import prepare_feature_store
from features.manifest import (save_manifest, load_manifest, assert_compatible,
                               with_training_params)
from baselines_ml.metrics import calc_metrics_numpy, measure_inference_time
from baselines_ml.es_split import split_train_es, ES_FRAC, ES_PROTOCOL
from baselines_ml.lgbm_baseline import LGBMBaseline
from baselines_ml.catboost_baseline import CatBoostBaseline
from baselines_ml.xgboost_baseline import XGBoostBaseline
from baselines_ml.tree_baselines import ExtraTreesBaseline
from baselines_ml.model_selection import select_champion_ml_model


MODEL_CLASSES = {
    'lightgbm': LGBMBaseline,
    'catboost': CatBoostBaseline,
    'xgboost': XGBoostBaseline,
    'extra_trees': ExtraTreesBaseline
}

# 3 mô hình GBDT cốt lõi (Histogram-based, tối ưu hóa tốc độ và độ chính xác trên dữ liệu lớn)
ALL_MODELS = ['lightgbm', 'catboost', 'xgboost']
ALL_DATASETS = ['sdn', 'geant', 'abilene']

# Tham số chỉ ảnh hưởng tới cách chạy (seed, số luồng CPU, mức log), không ảnh hưởng
# tới thí nghiệm. Loại khỏi model_config vì chúng khác nhau hợp lệ giữa các run.
_RUNTIME_ONLY_PARAMS = {'random_state', 'random_seed', 'seed', 'n_jobs',
                        'thread_count', 'verbose', 'verbosity'}


def get_model_instance(m_name, seed=42, quick_check=False):
    cls = MODEL_CLASSES[m_name.lower()]
    if quick_check:
        if m_name in ['lightgbm', 'xgboost']:
            return cls(n_estimators=20, random_state=seed)
        elif m_name == 'catboost':
            return cls(iterations=20, random_seed=seed)
        elif m_name == 'extra_trees':
            return cls(n_estimators=10, max_depth=6, random_state=seed)
    else:
        if m_name in ['lightgbm', 'xgboost', 'extra_trees']:
            return cls(random_state=seed)
        elif m_name == 'catboost':
            return cls(random_seed=seed)
    return cls()


def model_config_json(m_key, inst, quick_check):
    """Mô tả cấu hình thí nghiệm của một mô hình ML, ghi vào manifest.

    Gồm giao thức early stopping - thứ phân biệt run sinh ra TRƯỚC khi sửa lỗi B4
    (early stopping trên Val) với run sinh ra SAU (early stopping trên đuôi Train).
    Hai loại run đó không được trộn trong cùng một bảng kết quả.
    """
    params = {k: v for k, v in (getattr(inst, 'params', None) or {}).items()
              if k not in _RUNTIME_ONLY_PARAMS}
    cfg = {
        'model': m_key,
        'es_protocol': ES_PROTOCOL,
        'es_frac': ES_FRAC,
        'early_stopping_rounds': getattr(inst, 'early_stopping_rounds', None),
        'quick_check': bool(quick_check),
        'params': params,
    }
    return json.dumps(cfg, sort_keys=True, default=str)


def run_ml_experiments(models=None, datasets=None, runs=5, quick_check=False,
                       skip_existing=False, run_ids=None):
    if models is None or 'all' in models:
        models = ALL_MODELS
    if datasets is None or 'all' in datasets:
        datasets = ALL_DATASETS

    # run_ids cho phép chia việc giữa hai tài khoản Kaggle như nhánh học sâu.
    # Seed = 42 + run_id, nên run 7 luôn dùng seed 49 bất kể chạy ở đâu.
    run_ids = list(range(runs)) if run_ids is None else sorted(set(run_ids))
    full_coverage = run_ids == list(range(runs))

    results_dir = os.path.join(parent_dir, 'results')
    logs_dir = os.path.join(parent_dir, 'logs')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 80)
    print(" MODULE A: TRADITIONAL ML BASELINE BENCHMARK (SHARED MODEL)")
    print(f" Datasets: {datasets} | Models: {models} | Runs: {run_ids} | Quick Check: {quick_check}")
    print(f" Early stopping: {int(ES_FRAC * 100)}% cuối tập Train ({ES_PROTOCOL}) - KHÔNG dùng Val")
    print("=" * 80)

    val_records_for_champion = []

    for ds in datasets:
        print(f"\n---> Chuẩn bị dữ liệu cho Dataset: {ds.upper()}...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = prepare_feature_store(
            ds, data_dir=os.path.join(parent_dir, 'data')
        )
        n_flows = int(metadata['num_flows'])
        print(f"     Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape} | Features: {len(metadata['feature_names'])}")

        if quick_check:
            # Lấy tập mẫu nhỏ để kiểm tra nhanh luồng chạy. Số hàng Train làm tròn về
            # bội số của n_flows để việc tách Train-ES cắt đúng ranh giới bước thời gian.
            sample_tr = max(n_flows * 2, (min(5000, len(X_train)) // n_flows) * n_flows)
            sample_va = min(1000, len(X_val))
            sample_te = min(2000, len(X_test))
            X_tr, y_tr = X_train[:sample_tr], y_train[:sample_tr]
            X_va, y_va = X_val[:sample_va], y_val[:sample_va]
            X_te, y_te = X_test[:sample_te], y_test[:sample_te]
        else:
            X_tr, y_tr = X_train, y_train
            X_va, y_va = X_val, y_val
            X_te, y_te = X_test, y_test

        # Lỗi B4: trước đây fit(..., X_val=X_va, ...) - early stopping trên chính tập
        # Val mà bộ gộp dùng để khớp trọng số. Giờ tách đuôi Train làm tập early
        # stopping, áp dụng THỐNG NHẤT cho cả ba mô hình để bảng so sánh công bằng.
        X_fit, y_fit, X_es, y_es = split_train_es(X_tr, y_tr, n_flows)
        print(f"     Train-fit: {X_fit.shape[0]} hàng | Train-ES: {X_es.shape[0]} hàng "
              f"({X_es.shape[0] // n_flows} bước thời gian cuối)")

        for m_name in models:
            run_metrics = []
            m_key = m_name.lower().replace('-', '_')
            log_model_name = f"{m_key}_data_{ds}_shared"

            for run_id in run_ids:
                seed = 42 + run_id
                run_dir = os.path.join(logs_dir, log_model_name, f"run_{run_id}")
                os.makedirs(run_dir, exist_ok=True)
                test_csv = os.path.join(run_dir, 'test_metrics.csv')
                val_csv = os.path.join(run_dir, 'val_metrics.csv')
                model_file = os.path.join(run_dir, 'model.bin')

                model_inst = get_model_instance(m_key, seed=seed, quick_check=quick_check)
                expected_mf = with_training_params(
                    metadata['manifest'], seed=seed,
                    model_config=model_config_json(m_key, model_inst, quick_check))

                if skip_existing and os.path.exists(test_csv):
                    # Chỉ bỏ qua khi manifest khớp. Run sinh ra trước khi sửa lỗi B4
                    # không có manifest (hoặc khác es_protocol) -> huấn luyện lại, để
                    # không trộn hai giao thức early stopping trong một bảng kết quả.
                    try:
                        assert_compatible(expected_mf, load_manifest(run_dir))
                        reusable = True
                    except Exception as e:
                        reusable = False
                        print(f"  [RERUN] {m_name} {ds.upper()} run {run_id}: "
                              f"{str(e).splitlines()[0][:120]}", flush=True)

                    if reusable:
                        try:
                            prev_df = pd.read_csv(test_csv)
                            v_df = pd.read_csv(val_csv)
                            m_dict = prev_df.iloc[0].to_dict()
                            m_dict['run'] = run_id
                            run_metrics.append(m_dict)
                            val_records_for_champion.append({
                                'dataset': ds,
                                'model': m_key,
                                'run': run_id,
                                'val_mse': float(v_df.iloc[0]['mse']),
                                'inference_time_ms': float(v_df.iloc[0].get('inference_time_ms', 0.0))
                            })
                            print(f"  [SKIP] Đã có kết quả: {m_name} trên {ds.upper()} [run {run_id}]")
                            continue
                        except Exception as e:
                            print(f"    (Không dùng lại được kết quả cũ, huấn luyện lại: {e})")

                print(f"  [*] Huấn luyện {m_name.upper()} trên {ds.upper()} [run {run_id}] (Seed {seed})...", flush=True)
                t0 = time.time()
                try:
                    # Truyền theo VỊ TRÍ: XGBoostBaseline.fit nhận (X_es, y_es), các lớp
                    # khác nhận (X_val, y_val). Gọi bằng từ khoá X_val= từng làm XGBoost
                    # sập với TypeError.
                    model_inst.fit(X_fit, y_fit, X_es, y_es)
                except (ImportError, ModuleNotFoundError) as e:
                    print(f"  [WARN] Thư viện cho {m_name.upper()} chưa được cài đặt ({e}), bỏ qua mô hình này.", flush=True)
                    break
                fit_time = time.time() - t0

                # Val giờ là dữ liệu KHÔNG tham gia huấn luyện -> chọn champion công bằng
                val_preds = model_inst.predict(X_va)
                val_metrics = calc_metrics_numpy(val_preds, y_va)
                val_inf_time = measure_inference_time(lambda b: model_inst.predict(b), X_va, batch_size=64)
                val_metrics['inference_time_ms'] = val_inf_time

                val_records_for_champion.append({
                    'dataset': ds,
                    'model': m_key,
                    'run': run_id,
                    'val_mse': val_metrics['mse'],
                    'inference_time_ms': val_inf_time
                })

                test_metrics, test_preds = model_inst.evaluate(X_te, y_te, batch_size=64)
                test_metrics['run'] = run_id
                test_metrics['fit_time_s'] = fit_time
                run_metrics.append(test_metrics)

                pd.DataFrame([test_metrics]).to_csv(test_csv, index=False)
                pd.DataFrame([val_metrics]).to_csv(val_csv, index=False)
                np.save(os.path.join(run_dir, 'y_pred_data.npy'), test_preds)
                np.save(os.path.join(run_dir, 'y_real_data.npy'), y_te)
                try:
                    model_inst.save(model_file)
                except Exception as e:
                    print(f"    (Cảnh báo lưu checkpoint model: {e})")
                # Ghi manifest SAU CÙNG: có manifest nghĩa là run đã hoàn tất trọn vẹn.
                save_manifest(expected_mf, run_dir)

                print(f"      -> Test MSE={test_metrics['mse']*1000.0:.3f}e-3 | MAE={test_metrics['mae']*1000.0:.3f}e-3 | Inf Time={test_metrics['inference_time_ms']:.2f} ms | fit {fit_time:.0f}s")

            if full_coverage:
                df_runs = pd.DataFrame(run_metrics)
                out_csv = os.path.join(results_dir, f"results_{m_key}_data_{ds}.csv")
                df_runs.to_csv(out_csv, index=False)

    # Bảng tổng hợp và champion chỉ có nghĩa khi thấy ĐỦ mọi run và ĐỦ mọi mô hình ứng viên.
    #  - Chạy một phần run_ids (mỗi tài khoản Kaggle một nửa): mỗi bên sẽ ghi một bảng
    #    và một champion chỉ dựa trên nửa của mình -> conflict khi gộp, và hai champion
    #    có thể khác nhau.
    #  - Chạy thiếu mô hình (ví dụ --models lightgbm,catboost): champion bị chọn trong
    #    tập ứng viên thiếu XGBoost -> âm thầm thay nhánh ML của mô hình đề xuất.
    missing_models = [m for m in ALL_MODELS if m not in models]
    if not full_coverage:
        print(f"\n[CHƯA CHỌN CHAMPION] Mới chạy run {run_ids}, chưa đủ 0..{runs - 1}.")
        print("  Sau khi gộp nhánh của các tài khoản, chạy trên máy local (không huấn luyện lại):")
        print(f"    python -m baselines_ml.run_ml_baselines --datasets {','.join(datasets)} "
              f"--runs {runs} --skip_existing")
    elif missing_models:
        print(f"\n[CHƯA CHỌN CHAMPION] Thiếu mô hình ứng viên {missing_models}.")
        print("  Champion chỉ được chọn khi chạy đủ lightgbm, catboost và xgboost.")
    elif val_records_for_champion:
        champion_name, rank_table = select_champion_ml_model(val_records_for_champion, results_dir=results_dir)
        print("\nBẢNG XẾP HẠNG TRADITIONAL ML (TẬP VALIDATION):")
        print(rank_table.to_string(index=False))

    return val_records_for_champion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Traditional ML Baselines (Module A)")
    parser.add_argument('--models', type=str, default='all', help="Comma-separated models: lightgbm,catboost,xgboost (mặc định 'all' gồm 3 GBDT). Tùy chọn: extra_trees")
    parser.add_argument('--datasets', type=str, default='all', help="Comma-separated datasets: sdn,geant,abilene hoặc 'all'")
    parser.add_argument('--runs', type=int, default=5, help="Tổng số run của thí nghiệm (mặc định 5)")
    parser.add_argument('--run_ids', type=str, default=None,
                        help="Chỉ chạy các run này, ví dụ '0,1,2,3,4'. Dùng để chia việc giữa "
                             "hai tài khoản Kaggle. Khi chưa đủ 0..runs-1 thì không ghi bảng "
                             "tổng hợp và không chọn champion.")
    parser.add_argument('--quick_check', action='store_true', help="Chạy kiểm tra nhanh logic hệ thống")
    parser.add_argument('--skip_existing', action='store_true',
                        help="Bỏ qua run đã có kết quả VÀ có manifest khớp")

    args = parser.parse_args()

    m_list = [m.strip() for m in args.models.split(',')] if args.models != 'all' else ALL_MODELS
    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ALL_DATASETS
    rid = [int(x) for x in args.run_ids.split(',')] if args.run_ids else None

    run_ml_experiments(
        models=m_list,
        datasets=d_list,
        runs=args.runs,
        quick_check=args.quick_check,
        skip_existing=args.skip_existing,
        run_ids=rid,
    )
