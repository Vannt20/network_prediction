"""
Baseline tầm thường - PHẢI chạy TRƯỚC khi huấn luyện bất cứ mô hình nào.

Lý do đặt trước chứ không phải sau: nếu một nhánh học sâu không vượt được
persistence sau khi huấn luyện, đó là tín hiệu dừng ngay lập tức. Với Géant và
Abilene, nếu bước này có từ đầu thì đã phát hiện nhánh Local chết ngay ngày đầu.

Số tham chiếu đã đo (MSE x1e-3 trên Test):
    persistence: SDN 26.554 | Géant 0.855 | Abilene 2.363
Persistence tốt hơn GWN đã công bố trên Géant (0.855 vs 0.879) và tốt hơn 62% trên
Abilene (2.363 vs 6.220). Đây là thông tin bắt buộc phải nêu trong luận văn.
"""
import numpy as np


def persistence(x_last: np.ndarray) -> np.ndarray:
    """y_hat[t] = x[t-1]."""
    return np.asarray(x_last, dtype=np.float64)


def seasonal_naive(series: np.ndarray, idx: np.ndarray, period: int) -> np.ndarray:
    """y_hat[t] = x[t - period]. `idx` là chỉ số tuyệt đối của từng bước trong `series`."""
    series = np.asarray(series, dtype=np.float64)
    idx = np.asarray(idx)
    src = np.clip(idx - period, 0, len(series) - 1)
    return series[src]


def historical_average(train_norm: np.ndarray, tod_train: np.ndarray, dow_train: np.ndarray,
                       tod_q: np.ndarray, dow_q: np.ndarray, n_bins: int = 288) -> np.ndarray:
    """Trung bình theo (luồng, time-of-day, day-of-week) ước lượng trên Train.

    tod/dow ở thang [0, 1). Ô nào không có mẫu trong Train thì lùi về trung bình
    toàn cục của luồng đó.
    """
    train_norm = np.asarray(train_norm, dtype=np.float64)
    N = train_norm.shape[1]
    n_dow = 7

    def key(tod, dow):
        b = np.minimum((np.asarray(tod) * n_bins).astype(int), n_bins - 1)
        d = np.minimum((np.asarray(dow) * n_dow).astype(int), n_dow - 1)
        return d * n_bins + b

    k_tr = key(tod_train, dow_train)
    table = np.zeros((n_dow * n_bins, N))
    count = np.zeros((n_dow * n_bins, 1))
    np.add.at(table, k_tr, train_norm)
    np.add.at(count, k_tr, 1.0)
    glob = train_norm.mean(axis=0)
    filled = np.where(count > 0, table / np.maximum(count, 1.0), glob[None, :])
    return filled[key(tod_q, dow_q)]


def skill_score(mse_model: float, mse_persistence: float) -> float:
    """Skill = 1 - MSE_model / MSE_persistence.

    Đây mới là con số nói lên mô hình học được gì. Giá trị của mô hình cũ:
    SDN 0.838, Abilene 0.314, Géant 0.046 - cho thấy ngay SDN mới là tập có
    tín hiệu thật để học.
    """
    if mse_persistence <= 0:
        return float('nan')
    return 1.0 - mse_model / mse_persistence
