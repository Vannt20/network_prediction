"""
Kiểm định thống kê cho v7.

Thay giao thức cũ (9 x t-test ghép cặp trên 10 run trộn hai pipeline) vì:
  - LocalSpatialTCN có std = 0.000 qua 10 run -> giả định của t-test bị vi phạm.
  - Hiệu chỉnh đa kiểm định chỉ được nhắc ở phần thảo luận, không có trong bảng.
  - Không có kiểm định nào so sánh chuỗi sai số theo thời gian.
"""
import numpy as np


def wilcoxon_signed_rank(a, b):
    """Trả (statistic, p_value). Dùng thay t-test khi phương sai suy biến."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    if np.allclose(d, 0):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        return float('nan'), 1.0
    return float(stat), float(p)


def holm_bonferroni(pvalues):
    """Hiệu chỉnh Holm-Bonferroni. Trả mảng p đã hiệu chỉnh, cùng thứ tự đầu vào.

    Giá trị này phải được đưa THẲNG vào bảng kết quả dưới cột `p_holm`,
    không chỉ nhắc trong phần thảo luận.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def diebold_mariano(e1, e2, h: int = 1, power: int = 2):
    """Kiểm định Diebold-Mariano trên hai chuỗi sai số của cùng một chuỗi thời gian.

    e1, e2: sai số (y_hat - y) theo thời gian. Trả (DM_stat, p_value) hai phía.
    H0: hai mô hình có độ chính xác dự báo như nhau.

    Đây là kiểm định đúng chuẩn để so sánh hai mô hình dự báo, và không phụ thuộc
    số lần chạy lại - nên áp dụng được cả cho persistence và XGBoost vốn gần như
    tất định, những mô hình mà t-test trên 10 run không nói được gì.
    """
    from scipy.stats import norm
    e1 = np.asarray(e1, dtype=np.float64).ravel()
    e2 = np.asarray(e2, dtype=np.float64).ravel()
    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    s = gamma0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1]
        s += 2.0 * (1.0 - lag / h) * cov
    var = s / T
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var)
    return float(dm), float(2.0 * (1.0 - norm.cdf(abs(dm))))
