"""
Tách tập early stopping (Train-ES) cho nhánh ML - sửa lỗi B4.

Lỗi B4: các mô hình GBDT early-stop trên chính tập Validation mà bộ gộp dùng để
khớp trọng số, nên nhánh ML "đẹp giả" đúng ở nơi ra quyết định. Trên Abilene, XGBoost
tốt nhất trên Val (0.961e-3) nhưng tệ nhất trên Test (2.763e-3), và bộ gộp dồn 79%
trọng số cho nó.

Sửa: early stopping trên 15% CUỐI của tập Train. Tập Validation không còn tham gia
vào việc huấn luyện nhánh ML, chỉ dùng để chọn champion và khớp bộ gộp.
"""
import numpy as np

ES_FRAC = 0.15
ES_PROTOCOL = "train_tail_by_time"


def split_train_es(X, y, n_flows, es_frac=ES_FRAC):
    """Cắt đuôi tập Train theo THỜI GIAN, đúng tại ranh giới một bước thời gian.

    Ma trận tabular của feature_store xếp theo thời gian trước (hàng t*N + f), nên
    đuôi mảng chính là đuôi thời gian. Nhưng cắt theo số hàng có thể chia đôi một
    bước thời gian - một phần luồng vào tập khớp, phần còn lại vào tập early
    stopping. Hàm này làm tròn điểm cắt về bội số của n_flows để tránh điều đó.

    Không lấy ngẫu nhiên: dữ liệu là chuỗi thời gian, lấy ngẫu nhiên sẽ rò rỉ tương
    lai vào quá khứ.

    Trả (X_fit, y_fit, X_es, y_es).
    """
    n = len(X)
    if n_flows <= 0 or n % n_flows != 0:
        raise ValueError(
            f"Số hàng ({n}) không chia hết cho số luồng ({n_flows}) - ma trận tabular "
            f"không có dạng [T*N, D] như kỳ vọng.")
    n_steps = n // n_flows
    cut_t = int(round(n_steps * (1.0 - es_frac)))
    if not 1 <= cut_t < n_steps:
        raise ValueError(f"Không đủ bước thời gian để tách Train-ES (n_steps={n_steps}).")
    cut = cut_t * n_flows
    return X[:cut], y[:cut], X[cut:], y[cut:]
