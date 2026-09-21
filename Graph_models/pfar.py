"""
PFAR - Per-Flow Adaptive Affine Recalibrator.

Thay thế cổng softmax lồi (Graph_models/contextual_gate.py) trong luồng chính.

Lý do thay: tổ hợp lồi (softmax => w >= 0, sum(w) = 1) bị chặn trên bởi nhánh mạnh
nhất. Giải quy hoạch toàn phương trên đơn hình cho thấy trên SDN, bộ trọng số lồi
tốt nhất có thể là w = [0, 0, 1] cho MSE 5.509e-3 - tức đúng bằng XGBoost đơn lẻ.
Không cách tinh chỉnh nào thắng được; phải bỏ ràng buộc lồi.

PFAR gộp theo dạng affine riêng cho từng luồng OD:

    y_hat[t, f] = alpha[f] + sum_k beta[f, k] * y_hat_k[t, f]

Hai chế độ:
  - PFAROffline: khớp ridge một lần, cố định trong suốt tập Test.
  - PFAROnline : khởi tạo từ nghiệm offline, cập nhật bằng bình phương tối thiểu
                 đệ quy có hệ số quên. NHÂN QUẢ TUYỆT ĐỐI - dự báo tại t chỉ dùng
                 thông tin đến t-1.
"""
import numpy as np


def _design(P: np.ndarray) -> np.ndarray:
    """[M, N, K] -> [M, N, K+1] bằng cách nối cột hằng số 1 (hệ số chệch alpha)."""
    P = np.asarray(P, dtype=np.float64)
    return np.concatenate([P, np.ones(P.shape[:2] + (1,), dtype=np.float64)], axis=-1)


def _ridge_solve(S: np.ndarray, c: np.ndarray, n: float, ridge: float,
                 prior: np.ndarray = None) -> np.ndarray:
    """Giải hệ chuẩn tắc có chính quy hoá ridge TỶ LỆ VỚI VẾT ma trận.

    S: [N, D, D] tổng tích luỹ x x^T;  c: [N, D] tổng tích luỹ x*y;  n: số mẫu hiệu dụng.

    Hai chi tiết bắt buộc, thiếu một trong hai thì bộ gộp phân kỳ:
      (1) Chuẩn hoá theo n. Không có bước này S tích luỹ vô hạn và mất cân bằng
          thang đo giữa các luồng (Géant nổ lên 7.17e-3 thay vì 0.815e-3).
      (2) Ridge tỷ lệ với trace(A)/D, không phải hằng số. Géant có 529 luồng, nhiều
          luồng gần như bằng 0 nên ma trận suy biến; ridge hằng số thì hoặc quá mạnh
          với SDN hoặc quá yếu với Géant.

    `prior` [N, D] là ĐIỂM CO của chính quy hoá:

        min_w  ||Xw - y||^2 / n  +  d * ||w - prior||^2
        =>     (A + d I) w = c/n + d * prior

    Mặc định prior = 0 (ridge cổ điển, co về dự báo hằng số).

    Truyền prior = one-hot(nhánh đơn lẻ tốt nhất) thì khi d -> vô cùng, w -> prior,
    tức PFAR HỘI TỤ ĐÚNG VỀ nhánh đơn lẻ đó. Nhờ vậy nhánh đơn lẻ tốt nhất nằm
    TRONG họ nghiệm mà bước chọn siêu tham số đang dò, nên bộ gộp không thể thua nó
    một cách có hệ thống. Đây là cơ sở của cổng C8 và là cách duy nhất biến yêu cầu
    "phải tốt hơn mọi nhánh đơn lẻ" thành một bảo đảm về cấu trúc thay vì một hy vọng.
    """
    D = S.shape[-1]
    A = S / max(n, 1e-12)
    tr = np.trace(A, axis1=1, axis2=2) / D
    d = ridge * (tr + 1e-12)
    rhs = c / max(n, 1e-12)
    if prior is not None:
        rhs = rhs + d[:, None] * prior
    return np.linalg.solve(A + d[:, None, None] * np.eye(D), rhs[..., None])[..., 0]


def best_single_prior(P_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    """Dựng prior one-hot trỏ vào nhánh đơn lẻ tốt nhất trên KHỐI KHỚP.

    Xác định trên khối khớp (OOF hoặc Val), không bao giờ trên Test (I1).
    Trả [N, K+1]: hệ số 1.0 cho nhánh tốt nhất, 0 cho phần còn lại và cho chệch.
    """
    P_fit = np.asarray(P_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    M, N, K = P_fit.shape
    errs = ((P_fit - y_fit[..., None]) ** 2).mean(axis=(0, 1))   # [K]
    k_best = int(np.argmin(errs))
    prior = np.zeros((N, K + 1), dtype=np.float64)
    prior[:, k_best] = 1.0
    return prior


def uniform_prior(P_fit: np.ndarray) -> np.ndarray:
    """Dựng prior trung bình đều 1/K trên mọi nhánh, chệch bằng 0.

    Vì sao cần, bên cạnh best_single_prior. Mỏ neo one-hot chỉ hợp lý khi nhánh đơn
    lẻ tốt nhất thực sự là đích đáng co về. Đo trên kết quả 10 run của v7:

        SDN     : nhánh tốt nhất 5.555e-3  <  trung bình 1/3  9.675e-3  -> one-hot đúng
        Géant   : nhánh tốt nhất 0.918e-3  >  trung bình 1/3  0.738e-3  -> one-hot SAI
        Abilene : nhánh tốt nhất 2.792e-3  >  trung bình 1/3  1.915e-3  -> one-hot SAI

    Trên Géant/Abilene, rolling-origin chọn ridge = 10.0 (co rất mạnh) về đúng mỏ neo
    tệ hơn, nên bộ gộp kẹt giữa hai giá trị (0.894 và 2.609) thay vì tiến về phía
    trung bình. Không có prior đều thì cấu hình tốt nhất đã biết nằm NGOÀI họ nghiệm
    mà bước chọn siêu tham số dò được - đúng kiểu thiếu sót mà best_single_prior sinh
    ra để tránh, chỉ là trên một trục khác.

    Khi d -> vô cùng, w -> prior, tức PFAR hội tụ đúng về trung bình cộng 1/K.
    """
    P_fit = np.asarray(P_fit, dtype=np.float64)
    M, N, K = P_fit.shape
    prior = np.zeros((N, K + 1), dtype=np.float64)
    prior[:, :K] = 1.0 / float(K)
    return prior


PRIOR_KINDS = ('none', 'best_single', 'uniform')


def make_prior(kind, P_fit: np.ndarray, y_fit: np.ndarray):
    """Trả prior [N, K+1] theo `kind`, hoặc None khi không co về đâu cả.

    Nhận cả bool để tương thích ngược với tham số shrink_to_best_single cũ:
    True -> 'best_single', False -> 'none'.
    """
    if kind is True:
        kind = 'best_single'
    elif kind is False or kind is None:
        kind = 'none'
    if kind not in PRIOR_KINDS:
        raise ValueError(f"prior không hợp lệ: {kind!r}. Chọn một trong {PRIOR_KINDS}.")
    if kind == 'none':
        return None
    if kind == 'best_single':
        return best_single_prior(P_fit, y_fit)
    return uniform_prior(P_fit)


class PFAROffline:
    """Chế độ tĩnh: khớp một lần trên ma trận OOF (hoặc Val nếu chưa có OOF)."""

    def __init__(self, ridge: float = 1e-4, prior: str = 'best_single'):
        self.ridge = float(ridge)
        self.prior_kind = prior
        self.W = None       # [N, K+1]
        self.prior = None   # [N, K+1]

    def fit(self, P: np.ndarray, y: np.ndarray) -> "PFAROffline":
        X = _design(P)
        y = np.asarray(y, dtype=np.float64)
        M = X.shape[0]
        self.prior = make_prior(self.prior_kind, P, y)
        S = np.einsum('mnk,mnl->nkl', X, X)
        c = np.einsum('mnk,mn->nk', X, y)
        self.W = _ridge_solve(S, c, float(M), self.ridge, self.prior)
        return self

    def predict(self, P: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("PFAROffline chưa được fit.")
        return np.einsum('mnk,nk->mn', _design(P), self.W)

    @property
    def coef_(self) -> np.ndarray:
        """[N, K+1] - thay bảng trọng số cổng cũ trong phần phân tích diễn giải."""
        return self.W


class PFAROnline:
    """Chế độ trực tuyến: RLS có hệ số quên, neo mềm về nghiệm offline.

    Giả định triển khai: hệ thống đo được lưu lượng thật tại bước t ngay sau khi nó
    xảy ra, nên dùng được nó để hiệu chỉnh cho bước t+1. Giả định này PHẢI được nêu
    rõ trong phần phương pháp của luận văn, và kết quả phải luôn báo cáo song song
    với chế độ tĩnh.
    """

    def __init__(self, ridge: float = 1e-4, lam: float = 0.999, anchor: float = 0.1,
                 prior: str = 'best_single'):
        self.ridge = float(ridge)
        self.lam = float(lam)
        self.anchor = float(anchor)
        self.prior_kind = prior

    def run(self, P_fit: np.ndarray, y_fit: np.ndarray,
            P_stream: np.ndarray, y_stream: np.ndarray) -> np.ndarray:
        """Khởi tạo thống kê từ (P_fit, y_fit), rồi chạy trực tuyến trên chuỗi stream.

        Trả [M, N] dự báo. Tại mỗi bước t, hệ số được tính TRƯỚC khi quan sát y[t];
        cập nhật thống kê chỉ diễn ra SAU khi đã ghi dự báo.
        """
        Xf = _design(P_fit)
        Xs = _design(P_stream)
        yf = np.asarray(y_fit, dtype=np.float64)
        ys = np.asarray(y_stream, dtype=np.float64)
        M, N, D = Xs.shape

        S = np.einsum('mnk,mnl->nkl', Xf, Xf)
        c = np.einsum('mnk,mn->nk', Xf, yf)
        n = float(Xf.shape[0])

        prior = make_prior(self.prior_kind, P_fit, yf)
        w0 = _ridge_solve(S, c, n, self.ridge, prior)
        out = np.empty((M, N), dtype=np.float64)

        for t in range(M):
            w = (1.0 - self.anchor) * _ridge_solve(S, c, n, self.ridge, prior) + self.anchor * w0
            out[t] = np.einsum('nk,nk->n', Xs[t], w)
            # Cập nhật SAU khi đã dự báo -> không rò rỉ y[t] vào dự báo tại t.
            S = self.lam * S + np.einsum('nk,nl->nkl', Xs[t], Xs[t])
            c = self.lam * c + Xs[t] * ys[t][:, None]
            n = self.lam * n + 1.0

        return out
