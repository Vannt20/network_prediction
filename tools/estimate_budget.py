"""
Quy đổi thời gian epoch ĐO ĐƯỢC thành ngân sách GPU cho toàn bộ thực nghiệm.

    python tools/estimate_budget.py                       # đọc mọi train_metrics.csv có sẵn
    python tools/estimate_budget.py --accounts 2 --gpus 2 # kế hoạch 2 tài khoản x 2 GPU

Vì sao cần script này: bảng ngân sách trong quy trình v6 suy ra thời gian HUẤN
LUYỆN từ thời gian SUY DIỄN (STWaveFormer 3.2-3.6 ms/batch) rồi nhân hệ số. Hai
cách tính độc lập cho kết quả lệch nhau 25-50 lần (~2.6 h so với 41-127 h), nên
không cách nào dùng để lập kế hoạch. Script này chỉ dùng số đo trực tiếp từ cột
`epoch_time_s` trong train_metrics.csv.

Số epoch dùng để ngoại suy lấy từ log v6 thực tế (early stopping với patience=30).
Chúng là ƯỚC LƯỢNG: v7 đổi loss sang MSE và thêm warmup 15 epoch, nên độ dài run
có thể khác. Cột `epoch_giả_định` in ra để bạn tự điều chỉnh.
"""
import os
import sys
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        try:
            _s.reconfigure(encoding='utf-8')
        except Exception:
            pass
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Số epoch trung vị quan sát được trong logs/ của v6 (early stopping, patience=30).
EPOCHS_V6 = {
    ('sdn', 'stwaveformer'): 174,
    ('geant', 'stwaveformer'): 193,
    ('abilene', 'stwaveformer'): 54,
    ('sdn', 'localspatialtcn'): 137,
    ('geant', 'localspatialtcn'): 41,
    ('abilene', 'localspatialtcn'): 40,
}


def collect_epoch_times(logs_dir='logs'):
    """Gom epoch_time_s từ mọi train_metrics.csv. Bỏ epoch đầu (khởi động CUDA)."""
    rows = []
    for f in glob.glob(os.path.join(logs_dir, '*', 'run_*', 'train_metrics.csv')):
        parts = f.replace('\\', '/').split('/')
        model_dir = parts[-3]
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if 'epoch_time_s' not in df.columns or df.empty:
            continue
        t = df['epoch_time_s'].to_numpy(dtype=float)
        if len(t) > 1:
            t = t[1:]          # epoch 1 gồm chi phí khởi tạo CUDA/cudnn autotune
        name = model_dir.rsplit('_seq_', 1)[0]
        model, ds = name.split('_data_')
        rows.append({'dataset': ds, 'model': model,
                     'n_epoch_do': len(t), 's_per_epoch': float(np.median(t))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs_dir', default='logs')
    ap.add_argument('--runs', type=int, default=10, help="Số run mỗi (dataset, model)")
    ap.add_argument('--accounts', type=int, default=2)
    ap.add_argument('--gpus', type=int, default=2, help="Số GPU mỗi session")
    ap.add_argument('--quota_h', type=float, default=30.0, help="Quota giờ/tuần mỗi tài khoản")
    a = ap.parse_args()

    df = collect_epoch_times(a.logs_dir)
    if df.empty:
        print("Không tìm thấy cột 'epoch_time_s' trong train_metrics.csv nào.\n"
              "Hãy chạy cell hiệu chỉnh bằng code v7 mới nhất, ví dụ:\n"
              "  python run_experiments.py --dataset abilene --model STWaveFormer "
              "--runs 1 --epochs 3")
        return 1

    df = df.groupby(['dataset', 'model'], as_index=False).agg(
        s_per_epoch=('s_per_epoch', 'median'), n_epoch_do=('n_epoch_do', 'sum'))
    df['epoch_gia_dinh'] = [EPOCHS_V6.get((r.dataset, r.model), np.nan)
                            for r in df.itertuples()]
    df['gio_moi_run'] = df.s_per_epoch * df.epoch_gia_dinh / 3600.0
    df['gio_tong'] = df.gio_moi_run * a.runs

    print("=" * 84)
    print("SỐ ĐO TRỰC TIẾP (median epoch_time_s, đã bỏ epoch 1)")
    print("=" * 84)
    print(df[['dataset', 'model', 'n_epoch_do', 's_per_epoch',
              'epoch_gia_dinh', 'gio_moi_run', 'gio_tong']]
          .to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    known = df.dropna(subset=['epoch_gia_dinh'])
    missing = [f"{d}/{m}" for (d, m) in EPOCHS_V6
               if not ((df.dataset == d) & (df.model == m)).any()]

    total = float(known.gio_tong.sum())
    streams = max(1, a.accounts * a.gpus)
    print()
    print("=" * 84)
    print("NGÂN SÁCH")
    print("=" * 84)
    print(f"  Tổng GPU-giờ (tuần tự)          : {total:8.1f} h")
    print(f"  Số luồng GPU ({a.accounts} tài khoản x {a.gpus} GPU): {streams}")
    print(f"  Thời gian tường minh (lý tưởng) : {total / streams:8.1f} h")
    print(f"  Quota tiêu tốn / tài khoản      : {total / streams:8.1f} h "
          f"(quota tính theo giờ SESSION, không phải giờ-GPU)")
    print(f"  Quota khả dụng / tài khoản      : {a.quota_h:8.1f} h/tuần")

    if total / streams > a.quota_h:
        print(f"\n  VƯỢT QUOTA. Các lựa chọn cắt giảm:")
        print(f"    - Giảm còn 5 run       -> {total / streams / 2:.1f} h")
        ab = float(known[known.dataset == 'abilene'].gio_tong.sum())
        if ab:
            print(f"    - Hoãn Abilene         -> {(total - ab) / streams:.1f} h "
                  f"(Abilene chiếm {100 * ab / total:.0f}% chi phí)")
    else:
        print(f"\n  NẰM TRONG QUOTA, còn dư {a.quota_h - total / streams:.1f} h.")

    if missing:
        print(f"\n  CHƯA ĐO: {', '.join(missing)}")
        print("  Con số trên chỉ tính phần đã đo. Hãy chạy hiệu chỉnh cho các cấu")
        print("  hình còn thiếu trước khi tin vào tổng.")

    print("\n  Lưu ý: 'epoch_gia_dinh' lấy từ log v6. v7 đổi loss sang MSE và thêm")
    print("  warmup 15 epoch nên độ dài run thực tế có thể khác đáng kể.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
