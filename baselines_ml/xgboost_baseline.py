import os
import joblib
import numpy as np
from baselines_ml.metrics import calc_metrics_numpy, measure_inference_time


class XGBoostBaseline:
    """
    XGBoost Regressor theo chiến lược multi_output: shared_model:
    - 1 model duy nhất cho toàn bộ N luồng OD.
    - tree_method: hist (tăng tốc độ gấp 10-20 lần và tiết kiệm RAM).
    - Objective: reg:squarederror (Squared Error / MSE chuẩn mực tối ưu hồi quy lưu lượng).
    - flow_id ở cột 0 được phân chia theo ngưỡng số học (numerical split).

    I5 - tập early stopping phải TÁCH KHỎI tập khớp bộ gộp. Trước đây fit() mặc
    định early-stop trên chính tập Validation mà cổng dùng để học trọng số, khiến
    nhánh ML "đẹp giả" đúng ở nơi ra quyết định: trên Abilene nó tốt nhất trên Val
    (0.961e-3) nhưng tệ nhất trên Test (2.763e-3), và bộ gộp dồn 79% trọng số cho
    nhánh tệ nhất. Từ v7, gọi fit() với X_es lấy từ 15% cuối của Train (Train-ES).
    """
    def __init__(self, objective='reg:squarederror',
                 max_depth=6, learning_rate=0.05, n_estimators=1000,
                 subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30,
                 random_state=42, n_jobs=-1, device='cpu', **kwargs):
        self.params = {
            'objective': objective,
            'tree_method': 'hist',
            'device': device,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': n_jobs,
            **kwargs
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    @staticmethod
    def split_train_es(X_train, y_train, es_frac=0.15):
        """Cắt phần cuối của Train làm tập early stopping (Train-ES).

        Cắt theo trật tự thời gian ở CUỐI Train, không lấy ngẫu nhiên: dữ liệu là
        chuỗi thời gian nên lấy ngẫu nhiên sẽ rò rỉ tương lai vào quá khứ.
        Trả (X_fit, y_fit, X_es, y_es, idx_es) - idx_es phục vụ cổng C6.
        """
        n = len(X_train)
        cut = int(n * (1.0 - es_frac))
        idx_es = np.arange(cut, n)
        return X_train[:cut], y_train[:cut], X_train[cut:], y_train[cut:], idx_es

    def fit(self, X_train, y_train, X_es=None, y_es=None):
        """X_es/y_es: tập dùng RIÊNG cho early stopping.

        KHÔNG được truyền vào tập sẽ dùng để khớp bộ gộp (vi phạm I5).
        """
        import xgboost as xgb

        use_es = X_es is not None and y_es is not None and self.early_stopping_rounds
        self.model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds if use_es else None
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_es, y_es)] if use_es else None,
            verbose=False
        )
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, None)

    def evaluate(self, X_test, y_test, batch_size=64):
        preds = self.predict(X_test)
        metrics = calc_metrics_numpy(preds, y_test)
        inf_time = measure_inference_time(lambda b: self.predict(b), X_test, batch_size=batch_size)
        metrics['inference_time_ms'] = inf_time
        return metrics, preds

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)

    def load(self, filepath):
        import xgboost as xgb
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)
        return self
