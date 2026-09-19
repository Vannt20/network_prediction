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
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL_NAME_MAP = {
    'stwaveformer': 'ST-WaveFormer',
    'st_adaptive_ensemble': 'ST-Adaptive-Ensemble (Proposed)',
    'stadaptiveensemble': 'ST-Adaptive-Ensemble (Proposed)',
    'lightgbm': 'LightGBM (Module A)',
    'catboost': 'CatBoost (Module A)',
    'xgboost': 'XGBoost (Module A)',
    'extra_trees': 'Extra Trees (Module A)',
    'bigru': 'BiGRU',
    'gwn': 'GWN',
    'dcrnn': 'DCRNN'
}


def calculate_ci95(values):
    """
    Tính khoảng tin cậy 95% Confidence Interval.
    """
    n = len(values)
    if n < 2:
        return 0.0
    sem = stats.sem(values)
    h = sem * stats.t.ppf((1 + 0.95) / 2., n - 1)
    return h


def generate_thesis_report(results_dir=None, logs_dir=None):
    if results_dir is None:
        results_dir = os.path.join(parent_dir, 'results')
    if logs_dir is None:
        logs_dir = os.path.join(parent_dir, 'logs')

    os.makedirs(results_dir, exist_ok=True)
    all_records = []

    # 1. Thu thập từ results/*.csv
    csv_files = glob.glob(os.path.join(results_dir, "results_*_data_*.csv"))
    for f in csv_files:
        fname = os.path.basename(f)
        m = re.match(r"^results_(?P<model>.+)_data_(?P<dataset>[A-Za-z0-9]+)\.csv$", fname, re.IGNORECASE)
        if not m:
            continue
        raw_m = m.group('model')
        raw_ds = m.group('dataset').upper()

        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            df.columns = [c.strip().lower() for c in df.columns]
            
            clean_m = MODEL_NAME_MAP.get(raw_m.lower(), raw_m)
            
            mse_vals = df['mse'].values if 'mse' in df.columns else np.zeros(len(df))
            mae_vals = df['mae'].values if 'mae' in df.columns else np.zeros(len(df))
            rmse_vals = df['rmse'].values if 'rmse' in df.columns else np.sqrt(mse_vals)
            rse_vals = df['rse'].values if 'rse' in df.columns else np.zeros(len(df))
            mape_vals = df['mape'].values if 'mape' in df.columns else np.zeros(len(df))
            time_vals = df['inference_time_ms'].values if 'inference_time_ms' in df.columns else np.zeros(len(df))

            n_runs = len(df)
            mean_mse = np.mean(mse_vals) * 1000.0
            std_mse = np.std(mse_vals) * 1000.0
            ci_mse = calculate_ci95(mse_vals * 1000.0)

            mean_mae = np.mean(mae_vals) * 1000.0
            std_mae = np.std(mae_vals) * 1000.0
            ci_mae = calculate_ci95(mae_vals * 1000.0)

            all_records.append({
                'Dataset': raw_ds,
                'Model': clean_m,
                'Runs': n_runs,
                'MSE (x10^-3)': f"{mean_mse:.3f} ± {std_mse:.3f}",
                'MSE 95% CI': f"[{mean_mse - ci_mse:.3f}, {mean_mse + ci_mse:.3f}]",
                'MAE (x10^-3)': f"{mean_mae:.3f} ± {std_mae:.3f}",
                'MAE 95% CI': f"[{mean_mae - ci_mae:.3f}, {mean_mae + ci_mae:.3f}]",
                'RMSE': f"{np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}",
                'RSE': f"{np.mean(rse_vals):.4f}",
                'MAPE': f"{np.mean(mape_vals):.4f}",
                'Inference Time (ms)': f"{np.mean(time_vals):.2f} ± {np.std(time_vals):.2f}"
            })
        except Exception as e:
            print(f"Lỗi đọc {f}: {e}")

    summary_df = pd.DataFrame(all_records)
    if summary_df.empty:
        print("Không có kết quả nào để xuất báo cáo.")
        return summary_df

    # Sắp xếp thứ tự ưu tiên
    ds_order = {'SDN': 1, 'GEANT': 2, 'ABILENE': 3}
    summary_df['ds_rank'] = summary_df['Dataset'].map(lambda x: ds_order.get(x, 99))
    summary_df = summary_df.sort_values(by=['ds_rank', 'Model']).drop(columns=['ds_rank']).reset_index(drop=True)

    out_csv = os.path.join(results_dir, 'bang_tong_hop_luan_van.csv')
    out_xlsx = os.path.join(results_dir, 'bang_tong_hop_luan_van.xlsx')

    summary_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    try:
        summary_df.to_excel(out_xlsx, index=False)
    except Exception:
        pass

    print("\n" + "=" * 100)
    print(" BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐÁNH GIÁ MÔ HÌNH PHỤC VỤ LUẬN VĂN THẠC SĨ ")
    print(" (Báo cáo: Mean ± Std và 95% Confidence Interval qua 5 lần chạy độc lập)")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)
    print(f"-> Đã lưu bảng kết quả chuẩn tại: {out_csv}")
    print(f"-> Đã lưu file Excel tại: {out_xlsx}\n")

    return summary_df


if __name__ == '__main__':
    generate_thesis_report()
