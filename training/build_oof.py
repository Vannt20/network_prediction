"""
Sinh ma trận dự đoán out-of-fold (OOF) trên tập Train - giai đoạn Stage B.

Vì sao bắt buộc. Bộ gộp hiện học từ 10% dữ liệu nằm trong MỘT khoảng thời gian duy
nhất, nên nó học đặc tính của một chế độ vận hành rồi đem áp lên tập Test có chế độ
khác. MSE các nhánh trên Val so với Test:

    Géant  - Global   3.292 vs 0.801   (4.1x)
    Géant  - Local    4.949 vs 0.854   (5.8x)
    Abilene- ML       0.961 vs 2.763   (0.35x)

Hệ quả đo được trên Abilene: bộ gộp dồn 79% trọng số cho nhánh TỆ NHẤT trên Test.

Ngoài việc khớp tham số, OOF còn cần cho việc CHỌN SIÊU THAM SỐ. Đo trên cache
hiện có: trên Géant, khối chọn chỉ-có-Val cho tín hiệu phẳng (MSE rolling-origin
biến thiên 6% trong khi MSE trên Test biến thiên 300%), tức ridge không định danh
được từ Val. Quy tắc 1-SE giảm nhẹ triệu chứng nhưng chỉ khối OOF phủ nhiều chế độ
mới chữa được gốc.

    Train (70%)                                   Val(10%)  Test(20%)
    |- fold1 -|- fold2 -|- fold3 -|- fold4 -|- fold5 -|
               huấn luyện trên các fold TRƯỚC -> dự đoán fold hiện tại

Fold 1 bị bỏ (không có lịch sử). Phủ ~80% tập Train.

Chi phí: fold k huấn luyện trên (k-1)/K dữ liệu Train, tổng qua 5 fold là
(1+2+3+4)/5 = 2.0x một lần huấn luyện đầy đủ, cộng mô hình cuối trên toàn Train
= 3.0x tổng cộng (KHÔNG phải 5x).
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import asdict
from features.feature_store import prepare_feature_store, DATASET_CONFIGS
from features.manifest import save_manifest


def fold_edges(n: int, K: int = 5):
    """Biên các fold liên tiếp theo trật tự thời gian (blocked, không xáo trộn)."""
    return np.linspace(0, n, K + 1).astype(int)


def oof_plan(n_windows: int, K: int = 5):
    """Trả danh sách (k, chỉ số huấn luyện, chỉ số dự đoán) cho từng fold.

    Fold đầu tiên bị bỏ vì không có lịch sử để huấn luyện.
    """
    e = fold_edges(n_windows, K)
    plan = []
    for k in range(1, K):
        plan.append((k, np.arange(0, e[k]), np.arange(e[k], e[k + 1])))
    return plan


def build_oof_matrix(ds_key, run_id, K=5, cache_dir='cache', logs_dir='logs',
                     train_fn=None, predict_fn=None):
    """Dựng ma trận OOF cho một run.

    train_fn(ds_key, run_id, fold_k, idx_train) -> đối tượng mô hình ba nhánh
    predict_fn(models, idx_predict)             -> [len(idx), N, 3]

    Hai hàm này được truyền từ run_pipeline_v7.py để module này không phụ thuộc
    trực tiếp vào vòng huấn luyện, giúp kiểm thử được logic chia fold một cách
    độc lập.

    Ghi ra cache/{ds}_run_{r}_oof.pt gồm P_oof [M,N,3], y_oof [M,N], x_last, manifest.
    """
    if train_fn is None or predict_fn is None:
        raise NotImplementedError(
            "build_oof_matrix cần train_fn/predict_fn. Giai đoạn Stage B - chạy qua "
            "run_pipeline_v7.py --stage oof sau khi Stage A đã hoàn tất."
        )
    *_, meta = prepare_feature_store(ds_key)
    seq_len = DATASET_CONFIGS[ds_key]['seq_len']
    n_win = max(0, len(meta['train_norm']) - seq_len)

    P_parts, y_parts, idx_parts = [], [], []
    for k, idx_tr, idx_pr in oof_plan(n_win, K):
        print(f"  [OOF] {ds_key} run {run_id} fold {k}/{K - 1}: "
              f"huấn luyện trên {len(idx_tr)} cửa sổ, dự đoán {len(idx_pr)}", flush=True)
        models = train_fn(ds_key, run_id, k, idx_tr)
        P_parts.append(predict_fn(models, idx_pr))
        idx_parts.append(idx_pr)

    P_oof = np.concatenate(P_parts, axis=0)
    idx_all = np.concatenate(idx_parts, axis=0)
    train_norm = np.asarray(meta['train_norm'], dtype=np.float32)
    y_oof = train_norm[idx_all + seq_len]
    x_last = train_norm[idx_all + seq_len - 1]

    if not (len(P_oof) == len(y_oof) == len(x_last)):
        raise RuntimeError("OOF lệch kích thước - không được cắt ngầm (I2).")

    out = {
        'P_oof': torch.tensor(P_oof, dtype=torch.float32),
        'y_oof': torch.tensor(y_oof, dtype=torch.float32),
        'x_last': torch.tensor(x_last, dtype=torch.float32),
        'idx': torch.tensor(idx_all, dtype=torch.long),
        'manifest': asdict(meta['manifest']),
    }
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{ds_key}_run_{run_id}_oof.pt')
    with open(path, 'wb') as f:
        torch.save(out, f)
    print(f"  [OOF] Đã ghi {path} ({len(P_oof)} cửa sổ, phủ "
          f"{100.0 * len(P_oof) / max(n_win, 1):.0f}% tập Train)", flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()
    for r in range(a.runs):
        build_oof_matrix(a.dataset, r, K=a.folds)
