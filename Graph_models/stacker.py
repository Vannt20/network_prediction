"""
Bộ gộp hai tầng của v7 (sửa đổi S2 so với quy trình v6).

v6 giới hạn bộ gộp ở dạng affine. Đo trên cache hiện có cho thấy điều đó bỏ lỡ
khoảng 7% trên SDN:

    PFAR online (affine, v6)                 4.312e-3
    Stacker phi tuyến tĩnh, khớp trên Val    4.086e-3
    Stacker phi tuyến + PFAR online          3.997e-3

Nhưng trên Géant/Abilene, stacker phi tuyến khớp trên Val lại TỆ HẲN (Géant 1.229
so với 0.815) vì Val chỉ là một lát cắt 10% của một chế độ vận hành duy nhất. Đó là
lý do tầng phi tuyến bắt buộc phải khớp trên ma trận OOF, và là lý do có cổng C7:
nếu tầng phi tuyến không tự chứng minh được trên khối kiểm định thì tự động rút về
PFAR affine thuần.

Kiến trúc:
    Tầng 1 (tuỳ chọn): GBM nông trên [nhánh, persistence, hiệu số nhánh, ngữ cảnh, flow_id]
    Tầng 2 (luôn có) : PFAR online trên [nhánh..., đầu ra tầng 1]
"""
import numpy as np

from Graph_models.pfar import PFAROffline, PFAROnline


def build_stack_features(P: np.ndarray, x_last: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Dựng ma trận đặc trưng cho tầng phi tuyến.

    P       : [M, N, K] dự đoán các nhánh (thứ tự Global, Local, ML)
    x_last  : [M, N] giá trị bước t-1 (baseline persistence)
    context : [M, N, C] vector ngữ cảnh

    Trả [M*N, D]. Ngoài bản thân các dự đoán còn đưa vào hiệu số giữa các nhánh -
    độ bất đồng giữa các nhánh là chỉ báo kinh điển cho việc nên thận trọng, và là
    thông tin mà thiết kế cổng cũ hoàn toàn mù.
    """
    P = np.asarray(P, dtype=np.float32)
    x_last = np.asarray(x_last, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    M, N, K = P.shape

    parts = [P[..., k] for k in range(K)]
    parts.append(x_last)
    for k in range(K):
        parts.append(P[..., k] - x_last)
    for a in range(K):
        for b in range(a + 1, K):
            parts.append(P[..., a] - P[..., b])
    parts.append(P.std(axis=-1))
    for c in range(context.shape[-1]):
        parts.append(context[..., c])
    parts.append(np.tile(np.arange(N, dtype=np.float32), (M, 1)))   # flow_id

    return np.stack(parts, axis=-1).reshape(M * N, len(parts))


class NonlinearStacker:
    """Tầng 1: GBM nông. Cố ý giữ nông (max_depth <= 6) vì đầu vào đã là dự đoán."""

    def __init__(self, n_estimators=400, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42):
        self.kw = dict(n_estimators=n_estimators, max_depth=max_depth,
                       learning_rate=learning_rate, subsample=subsample,
                       colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                       random_state=random_state, tree_method='hist',
                       objective='reg:squarederror', verbosity=0)
        self.model = None

    def fit(self, P, x_last, context, y):
        import xgboost as xgb
        F = build_stack_features(P, x_last, context)
        self.model = xgb.XGBRegressor(**self.kw)
        self.model.fit(F, np.asarray(y, dtype=np.float32).reshape(-1))
        return self

    def predict(self, P, x_last, context) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NonlinearStacker chưa được fit.")
        F = build_stack_features(P, x_last, context)
        return self.model.predict(F).reshape(P.shape[0], P.shape[1]).astype(np.float64)


def _mse(a, y):
    return float(((np.asarray(a, np.float64) - np.asarray(y, np.float64)) ** 2).mean())


class TwoStageCombiner:
    """Bộ gộp v7 đầy đủ, kèm cổng C7.

    fit() nhận hai khối tách biệt:
      - (fit_*)   khối khớp tham số: ma trận OOF, hoặc Val khi chưa có OOF.
      - (sel_*)   khối kiểm định để quyết định GIỮ hay BỎ tầng phi tuyến (cổng C7).
                  Không bao giờ là tập Test.
    """

    def __init__(self, ridge=1e-4, lam=0.999, anchor=0.1, use_nonlinear=True,
                 stacker_kwargs=None):
        self.ridge = ridge
        self.lam = lam
        self.anchor = anchor
        self.use_nonlinear = use_nonlinear
        self.stacker_kwargs = stacker_kwargs or {}
        self.stacker = None
        self.c7_passed = None
        self.c7_detail = {}

    def fit(self, P_fit, x_fit, ctx_fit, y_fit, P_sel, x_sel, ctx_sel, y_sel):
        self.stacker = None
        self.c7_passed = False
        if not self.use_nonlinear:
            self.c7_detail = {'reason': 'tầng phi tuyến bị tắt theo cấu hình'}
            return self

        st = NonlinearStacker(**self.stacker_kwargs).fit(P_fit, x_fit, ctx_fit, y_fit)
        s_sel = st.predict(P_sel, x_sel, ctx_sel)

        # So sánh trên khối kiểm định: có tầng phi tuyến vs PFAR affine thuần.
        affine = PFAROffline(self.ridge).fit(P_fit, y_fit)
        mse_affine = _mse(affine.predict(P_sel), y_sel)
        aug_fit = np.concatenate([P_fit, st.predict(P_fit, x_fit, ctx_fit)[..., None]], axis=-1)
        aug_sel = np.concatenate([P_sel, s_sel[..., None]], axis=-1)
        mse_two = _mse(PFAROffline(self.ridge).fit(aug_fit, y_fit).predict(aug_sel), y_sel)

        self.c7_detail = {'mse_affine_only': mse_affine, 'mse_two_stage': mse_two}
        # Yêu cầu cải thiện thực chất >=1%, tránh giữ một tầng chỉ vì nhiễu.
        if mse_two < 0.99 * mse_affine:
            self.stacker = st
            self.c7_passed = True
        else:
            self.c7_detail['reason'] = 'tầng phi tuyến không cải thiện >=1% trên khối kiểm định -> rút về PFAR affine'
        return self

    def _augment(self, P, x_last, ctx):
        if self.stacker is None:
            return np.asarray(P, dtype=np.float64)
        s = self.stacker.predict(P, x_last, ctx)
        return np.concatenate([np.asarray(P, np.float64), s[..., None]], axis=-1)

    def predict_static(self, P_fit, x_fit, ctx_fit, y_fit, P_te, x_te, ctx_te):
        A_fit = self._augment(P_fit, x_fit, ctx_fit)
        A_te = self._augment(P_te, x_te, ctx_te)
        return PFAROffline(self.ridge).fit(A_fit, y_fit).predict(A_te)

    def predict_online(self, P_fit, x_fit, ctx_fit, y_fit, P_te, x_te, ctx_te, y_te):
        A_fit = self._augment(P_fit, x_fit, ctx_fit)
        A_te = self._augment(P_te, x_te, ctx_te)
        return PFAROnline(self.ridge, self.lam, self.anchor).run(A_fit, y_fit, A_te, y_te)
