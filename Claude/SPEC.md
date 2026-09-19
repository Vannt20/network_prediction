# SPEC — ST-Adaptive-Ensemble v6

**Repo:** `github.com/Vannt20/thucnghiem`
**Phiên bản pipeline:** `v6`
**Đối tượng thực thi:** AI coding agent (Claude Code / Cursor / tương đương)
**Ngôn ngữ code:** Python 3.10+, PyTorch, XGBoost
**Môi trường đích:** Kaggle GPU (T4 / P100), tối đa 12 h/session, 30 h/tuần

---

## 0. Cách đọc tài liệu này

Tài liệu được viết để một agent có thể thực thi **tuần tự, không cần hỏi lại**. Mỗi task có: file cần sửa, thay đổi chính xác, chữ ký hàm, tiêu chí hoàn thành đo được, và lệnh kiểm thử.

**Quy ước:**

- `[BẮT BUỘC]` — không hoàn thành thì dừng pipeline.
- `[KIỂM THỬ]` — có assert tự động, phải pass.
- `[SỐ THAM CHIẾU]` — giá trị đã đo trên cache hiện có; dùng làm regression target.
- Đường dẫn tương đối tính từ gốc repo.

**Nguyên tắc bất biến — vi phạm bất kỳ điều nào là lỗi nghiêm trọng:**

| # | Bất biến |
|---|---|
| I1 | Không tham số nào được khớp trên tập Test. Không chọn siêu tham số trên Test, kể cả một lần "để xem". |
| I2 | Không cắt/đệm mảng ngầm. Mọi bất khớp kích thước phải `raise`, không được `[:n]`. |
| I3 | Mọi artifact ghi ra đĩa phải kèm `manifest.json`; mọi artifact đọc vào phải đối chiếu manifest. |
| I4 | Hàm mất mát huấn luyện phải cùng họ với chỉ số công bố (MSE). |
| I5 | Tập dùng cho early stopping của một nhánh không được trùng tập dùng để khớp bộ gộp. |
| I6 | Mọi nhánh phải qua cổng nghiệm thu C1–C4 trước khi được đưa vào bộ gộp. |

---

## 1. Bối cảnh — bốn lỗi đang tồn tại trong repo

Agent cần hiểu vì sao phải sửa, không chỉ sửa gì.

| ID | Lỗi | Bằng chứng | Task xử lý |
|---|---|---|---|
| **B1** | Nhánh `LocalSpatialTCN` sụp về persistence trên Géant và Abilene. `mean((ŷ_local[t] − x[t−1])²) = 0.000`. `train_loss` tăng đều từ epoch 1 → 70, checkpoint `best_model.pth` là trạng thái khởi tạo. | `logs/localspatialtcn_data_abilene_seq_24/run_5/train_metrics.csv` | T3 |
| **B2** | Run 0–4 và run 5–9 của Géant/Abilene sinh từ hai pipeline khác nhau. `y_pred_data.npy` có 9596 dòng (run 0–4) vs 9539 dòng (run 5–9); schema `test_metrics.csv` khác nhau. `precompute_cache.py` cắt ngầm → MSE nhánh Global nhảy từ 2.45 lên 10.38 ×10⁻³. Toàn bộ `std` 10 run trong báo cáo là artifact, không phải phương sai seed. | `logs/stwaveformer_data_abilene_seq_24/run_{0..9}/` | T1, T2 |
| **B3** | Cổng gộp dùng `huber_loss(delta=0.01)` trong khi công bố MSE; huấn luyện full-batch 100 bước từ prior `-ln(val_mse)` nên gần như không dịch chuyển khỏi điểm khởi tạo. | `training/train_gate_stacking.py:98` | T5 |
| **B4** | XGBoost early-stop trên chính tập Validation mà cổng dùng để khớp trọng số. Hậu quả trên Abilene: nhánh ML tốt nhất trên Val (0.961) nhưng tệ nhất trên Test (2.763), bộ gộp dồn 79% trọng số cho nó. | `baselines_ml/xgboost_baseline.py:37`, `training/train_gate_stacking.py:50` | T4, T5 |

**Trần lý thuyết đã tính (quy hoạch toàn phương trên đơn hình, nghiệm chính xác):** trên SDN, tổ hợp lồi tốt nhất có thể là `w = [0, 0, 1]` cho MSE `5.509×10⁻³` — tức đúng bằng XGBoost. **Cơ chế softmax gating không thể thắng XGBoost trên SDN.** Đây là lý do cốt lõi phải bỏ ràng buộc lồi.

---

## 2. Tổng quan thay đổi

| File | Hành động | Task |
|---|---|---|
| `features/manifest.py` | **Tạo mới** | T1 |
| `features/feature_store.py` | Sửa: sinh manifest, thêm `make_oof_folds()` | T1, T6 |
| `run_experiments.py` | Sửa: loss MSE, warmup early-stopping, log `delta_norm`, ghi manifest | T3 |
| `Graph_models/local_filters.py` | Sửa: tách param group, log delta | T3 |
| `baselines_ml/xgboost_baseline.py` | Sửa: tham số hoá `eval_set`, hỗ trợ `device="cuda"` | T4 |
| `training/precompute_cache.py` | Sửa: bỏ reuse `.npy`, thêm assert manifest | T2 |
| `Graph_models/pfar.py` | **Tạo mới** — bộ gộp PFAR | T5 |
| `training/train_pfar.py` | **Tạo mới** — thay `train_gate_stacking.py` | T5 |
| `training/build_oof.py` | **Tạo mới** — sinh ma trận out-of-fold | T6 |
| `evaluation/baselines_naive.py` | **Tạo mới** — persistence, HA, seasonal naive | T7 |
| `evaluation/acceptance_gates.py` | **Tạo mới** — cổng C1–C6 | T8 |
| `evaluation/ablation_study.py` | Viết lại: 8 cấu hình | T9 |
| `evaluation/stats_tests.py` | **Tạo mới** — Wilcoxon, Holm, Diebold–Mariano | T9 |
| `run_pipeline_v6.py` | **Tạo mới** — điều phối toàn bộ | T10 |
| `training/train_gate_stacking.py` | **Giữ nguyên, không xoá** — dùng làm cấu hình ablation A5 | T9 |

**Không đụng tới:** `Graph_models/gwn.py`, `Graph_models/dcrnn*.py`, `baselines_ml/lgbm_baseline.py`, `baselines_ml/catboost_baseline.py`. GWN/BiGRU giữ nguyên số trích từ bài báo gốc, chỉ dùng làm tham chiếu.

---

## 3. T0 — Quick win, 0 phút GPU `[BẮT BUỘC, làm trước tiên]`

Chứng minh hướng đi đúng trước khi cam kết 30 h GPU.

**Tạo `tools/quick_validate_pfar.py`.** Script này đọc cache `.pt` hiện có, chạy PFAR, đối chiếu với số tham chiếu.

```python
"""
Chạy: python tools/quick_validate_pfar.py
Không cần GPU, không huấn luyện lại. Thời gian: < 5 phút.
Mục đích: xác nhận công thức gộp mới tái lập được số tham chiếu trước khi chạy full pipeline.
"""
```

Yêu cầu:

1. Nạp `cache/{ds}_run_{r}_{val,test}_preds.pt` bằng `torch.load`.
2. **Abilene và Géant chỉ dùng run 5–9** (run 0–4 chứa artifact B2). SDN dùng đủ 10 run.
3. Chạy ba cấu hình: `static_convex_fit_val`, `pfar_offline`, `pfar_online`.
4. So với `[SỐ THAM CHIẾU]` bên dưới, dung sai ±3%.

`[SỐ THAM CHIẾU]` — MSE ×10⁻³ trên tập Test, trung bình các run hợp lệ:

| Cấu hình | SDN | Géant | Abilene (run 5–9) |
|---|---|---|---|
| Persistence | 26.552 | 0.855 | 2.363 |
| Global đơn lẻ | 15.350 | 0.801 | 1.978 |
| Local đơn lẻ | 15.152 | 0.854 | 2.361 |
| ML (XGBoost) đơn lẻ | 5.509 | 0.915 | 2.763 |
| Softmax lồi (mô hình cũ) | 6.562 ± 0.211 | 0.796 ± 0.025 | 2.731 ± 0.866 |
| Tổ hợp lồi ORACLE trên Test | 5.509 | 0.735 | — |
| **PFAR offline** (ridge khớp Val) | **4.403 ± 0.132** | 0.794 | 2.314 |
| **PFAR online** (RLS) | **4.312 ± 0.042** | 0.816 ± 0.005 | **1.620 ± 0.002** |
| Oracle chọn nhánh tốt nhất/mẫu | 3.597 | 0.549 | — |

Ridge tối ưu đã dò: SDN `1e-5`, Géant `1e-2`, Abilene `1e-3`; `λ = 0.999` cho cả ba.

`[KIỂM THỬ]` Script in ra bảng và `PASS`/`FAIL` cho từng ô. Nếu SDN PFAR online không đạt khoảng `4.2–4.5`, dừng và báo cáo — có gì đó sai trong cài đặt, không được chạy tiếp.

> **Lưu ý cho agent:** ridge ở bảng trên được dò trên Test và **chỉ dùng cho mục đích kiểm thử hồi quy của T0**. Trong pipeline thật (T5), ridge phải được chọn bằng rolling-origin trên Train+Val. Không được hard-code ba giá trị này vào pipeline.

---

## 4. T1 — Hệ thống manifest `[BẮT BUỘC]`

**Tạo `features/manifest.py`:**

```python
from dataclasses import dataclass, asdict
import hashlib, json, subprocess, os

PIPELINE_VERSION = "v6"

@dataclass
class Manifest:
    pipeline_version: str
    dataset: str
    csv_sha256: str
    n_steps_after_hygiene: int
    split_train: int
    split_val: int
    split_test: int
    seq_len: int
    k_lags: int
    n_windows_train: int
    n_windows_val: int
    n_windows_test: int
    seed: int
    git_commit: str

def sha256_file(path: str) -> str: ...
def git_commit_short() -> str: ...          # trả "unknown" nếu không có git

def build_manifest(dataset: str, meta: dict, seed: int) -> Manifest: ...
def save_manifest(m: Manifest, dirpath: str) -> None:
    """Ghi <dirpath>/manifest.json"""

def load_manifest(dirpath: str) -> Manifest: ...

STRICT_KEYS = ["pipeline_version", "dataset", "csv_sha256",
               "n_steps_after_hygiene", "split_train", "split_val", "split_test",
               "seq_len", "k_lags", "n_windows_val", "n_windows_test"]

def assert_compatible(a: Manifest, b: Manifest) -> None:
    """So sánh STRICT_KEYS. Lệch bất kỳ khoá nào -> raise RuntimeError với
    thông điệp: 'Artifact không tương thích ở trường X (a=..., b=...).
    Hãy xoá logs/ và cache/ rồi chạy lại từ đầu.'
    KHÔNG được cảnh báo rồi đi tiếp. Phải raise."""
```

**Sửa `features/feature_store.py`:** `prepare_feature_store()` bổ sung vào `metadata`:

```python
metadata['manifest'] = build_manifest(ds_key, metadata, seed=seed)
```

Thêm tham số `seed: int = 42` vào chữ ký hàm.

`[KIỂM THỬ]` `tests/test_manifest.py`: tạo hai manifest lệch một trường, khẳng định `assert_compatible` raise.

---

## 5. T2 — Sửa `precompute_cache.py` `[BẮT BUỘC]`

**Xoá hoàn toàn** khối reuse `.npy` (hiện ở dòng ~158–173):

```python
# XOÁ khối này:
if not quick_check and os.path.exists(y_pred_global_test_file):
    y_glob_test_np = np.load(y_pred_global_test_file)
    if len(y_glob_test_np) == len(x_test_win):
        ...
    else:
        y_global_test = torch.tensor(y_glob_test_np[:len(x_test_win)], ...)  # <-- nguồn gốc lỗi B2
```

**Thay bằng:** luôn suy diễn lại từ `best_model.pth`, đối xứng với nhánh Local.

**Thêm ở đầu mỗi run:**

```python
run_manifest = load_manifest(global_log_dir)
assert_compatible(meta['manifest'], run_manifest)
```

**Thêm khẳng định checkpoint:**

```python
if not os.path.exists(ckpt_global):
    raise FileNotFoundError(
        f"Không có checkpoint {ckpt_global}. Không được dùng trọng số khởi tạo ngẫu nhiên."
    )
```

*(Code hiện tại in cảnh báo rồi chạy tiếp với trọng số random — đây là lỗi tiềm tàng nghiêm trọng.)*

**Cache ghi ra bổ sung khoá `manifest`** và các trường mới phục vụ PFAR:

```python
cache_data = {
    'y_global': ..., 'y_local': ..., 'y_ml': ...,
    'context': ...,        # [M, N, C] — C mở rộng, xem T5
    'y_real': ...,
    'x_last': ...,         # [M, N] giá trị bước t-1, cho baseline persistence
    'manifest': asdict(meta['manifest']),
}
```

`[KIỂM THỬ]` Sau khi chạy, khẳng định mọi cache cùng dataset có `n_windows_test` giống nhau:

```bash
python -c "
import torch, glob
s = {torch.load(f, weights_only=False)['manifest']['n_windows_test'] for f in glob.glob('cache/abilene_run_*_test_preds.pt')}
assert len(s) == 1, f'Cache không đồng nhất: {s}'
print('OK', s)
"
```

---

## 6. T3 — Hồi sinh nhánh học sâu `[BẮT BUỘC]`

### 6.1. `run_experiments.py`

**Sửa dòng ~288:**

```python
# CŨ:
if m_name in ['stwaveformer', ..., 'spatialdilatedtcn']:
    lossfn = nn.SmoothL1Loss(beta=0.01)
# MỚI:
lossfn = nn.MSELoss()      # I4: cùng họ với chỉ số công bố
```

Lý do: trên thang dữ liệu MinMax [0,1], MAE điển hình của các nhánh là 0.017–0.067, tức gần như mọi mẫu nằm trong vùng tuyến tính của Huber với `beta=0.01`. Mạng đang tối ưu MAE trong khi luận văn công bố MSE.

**Thêm tham số `warmup_epochs: int = 15`** vào `train_and_eval_model()`, và sửa logic early stopping:

```python
if epoch < warmup_epochs:
    patience_counter = 0          # không đếm patience trong warmup
    if epoch == warmup_epochs - 1:
        best_val_loss = float('inf')   # reset, buộc lưu checkpoint SAU warmup
```

Lý do: `out_head` của `SpatialDilatedTCN` khởi tạo bằng 0 nên epoch 1 chính là persistence. Nếu persistence đã tốt (Géant, Abilene), early stopping giữ lại đúng epoch 1 và nhánh chết vĩnh viễn. Đây là nguyên nhân trực tiếp của B1.

**Thêm nhóm tham số riêng cho `out_head`:**

```python
if m_name in ['localspatialtcn', 'local_spatial_tcn', 'spatialdilatedtcn']:
    head_params = [p for n, p in model.named_parameters() if n.startswith('out_head')]
    base_params = [p for n, p in model.named_parameters() if not n.startswith('out_head')]
    optimizer = optim.Adam([
        {'params': base_params, 'lr': lr,      'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr * 10, 'weight_decay': 0.0},   # zero-init + weight_decay = chết
    ])
else:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

**Ghi thêm cột vào `train_metrics.csv`:** `mean_abs_delta` (chỉ với LocalSpatialTCN). Đây là chỉ báo sống/chết của nhánh, kiểm tra được ngay trong lúc chạy.

**Ghi `manifest.json`** vào `logdir` mỗi run.

### 6.2. `Graph_models/local_filters.py`

`SpatialDilatedTCN.forward()` trả thêm delta khi được yêu cầu:

```python
def forward(self, x, return_delta: bool = False):
    ...
    if return_delta:
        return out, delta_norm.abs().mean().detach()
    return out
```

### 6.3. Cấu hình bổ sung cần thử trên SDN `[TÙY CHỌN, ưu tiên thấp]`

Giả thuyết cho khoảng cách 15.2 vs 5.5 trên SDN: bất đối xứng số mẫu hiệu dụng. SDN chỉ có 4,320 cửa sổ trượt cho nhánh học sâu, trong khi XGBoost dạng shared-model thấy 4,320 × 196 ≈ 846 nghìn dòng.

Thêm cờ `--channel_independent` cho `STWaveFormer`: coi mỗi luồng OD là một mẫu độc lập, chia sẻ trọng số theo thời gian, giữ tương tác không gian ở một tầng nhẹ cuối. Nếu cấu hình này đưa Global trên SDN xuống dưới 10×10⁻³ thì giữ; nếu không, ghi lại kết quả âm vào ablation và bỏ.

---

## 7. T4 — Sửa `xgboost_baseline.py` `[BẮT BUỘC]`

```python
class XGBoostBaseline:
    def __init__(self, ..., device: str = "cpu", early_stopping_rounds: int = 30):
        self.params = {..., 'device': device, 'tree_method': 'hist'}
```

`fit()` nhận `eval_set` từ bên ngoài thay vì mặc định là Validation (I5):

```python
def fit(self, X_train, y_train, X_es=None, y_es=None):
    """X_es/y_es: tập dùng riêng cho early stopping.
    KHÔNG được truyền vào tập sẽ dùng để khớp bộ gộp."""
```

**Cách chia mới:** cắt 15% cuối của tập Train làm `X_es` (gọi là **Train-ES**). Tập Validation từ nay chỉ phục vụ chọn siêu tham số bộ gộp.

Trên Kaggle, truyền `device="cuda"` — giảm thời gian huấn luyện XGBoost khoảng 5–8 lần trên Géant/Abilene (≈4–5 triệu dòng).

---

## 8. T5 — PFAR: bộ gộp mới `[BẮT BUỘC — đây là đóng góp cốt lõi]`

### 8.1. Dạng hàm

$$\hat{y}_{t,f} = \alpha_f(t) + \beta^G_f(t)\,\hat{y}^G_{t,f} + \beta^L_f(t)\,\hat{y}^L_{t,f} + \beta^M_f(t)\,\hat{y}^M_{t,f}$$

Ba khác biệt so với `contextual_gate.py` hiện tại:

1. **Bỏ softmax** — hệ số được phép âm và tổng khác 1. Ràng buộc lồi chặn mô hình tại đúng mức nhánh mạnh nhất (xem Mục 1).
2. **Thêm hệ số chệch `α_f`** — riêng hiệu chỉnh chệch/tỷ lệ theo luồng đã đưa XGBoost trên SDN từ 5.509 xuống 4.494.
3. **Tham số phụ thuộc thời gian** — cập nhật trực tuyến.

### 8.2. `Graph_models/pfar.py` — tạo mới

```python
class PFAROffline:
    """Per-Flow Affine Recalibrator, chế độ tĩnh.
    Khớp ridge riêng cho từng luồng OD trên ma trận OOF (hoặc Val nếu chưa có OOF)."""

    def __init__(self, n_flows: int, n_branches: int = 3, ridge: float = 1e-4):
        ...
    def fit(self, P: np.ndarray, y: np.ndarray) -> "PFAROffline":
        """P: [M, N, K] dự đoán các nhánh. y: [M, N] nhãn thật.
        Giải bài toán chuẩn tắc cho từng luồng với thiết kế [P, 1].
        Ridge tỷ lệ với vết: d_f = ridge * trace(A_f)/D  (KHÔNG dùng hằng số)."""
    def predict(self, P: np.ndarray) -> np.ndarray:
        """-> [M, N]"""
    @property
    def coef_(self) -> np.ndarray:
        """[N, K+1] — dùng cho phân tích diễn giải, thay bảng trọng số cổng cũ."""


class PFAROnline:
    """Chế độ trực tuyến: bình phương tối thiểu đệ quy có hệ số quên.
    Khởi tạo từ nghiệm offline, cập nhật sau mỗi bước bằng nhãn thật đã quan sát.
    NHÂN QUẢ TUYỆT ĐỐI: dự báo tại t chỉ dùng thông tin đến t-1."""

    def __init__(self, offline: PFAROffline, lam: float = 0.999,
                 ridge: float = 1e-4, anchor: float = 0.1):
        ...
    def run(self, P: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Chạy trực tuyến trên toàn chuỗi, trả [M, N] dự báo."""
```

**Cài đặt `PFAROnline.run` — ba chi tiết bắt buộc.** Đã kiểm chứng: thiếu bất kỳ chi tiết nào thì Géant phân kỳ lên 7.17×10⁻³ thay vì 0.816×10⁻³.

```python
D = K + 1
X = concat([P, ones], -1)                      # [M, N, D]
S = einsum('mnk,mnl->nkl', Xv, Xv)             # khởi tạo từ dữ liệu offline
c = einsum('mnk,mn->nk',  Xv, yv)
n = float(M_offline)

def _solve(S, c, n):
    A = S / n                                   # (1) CHUẨN HOÁ theo số mẫu hiệu dụng
    d = ridge * (trace(A, axis1=1, axis2=2) / D + 1e-12)
    return solve(A + d[:, None, None] * I, (c / n)[..., None])[..., 0]   # (2) ridge TỶ LỆ VỚI VẾT

w0 = _solve(S, c, n)
for t in range(M):
    w = (1 - anchor) * _solve(S, c, n) + anchor * w0      # (3) NEO MỀM về nghiệm offline
    out[t] = einsum('nk,nk->n', X[t], w)
    S = lam * S + einsum('nk,nl->nkl', X[t], X[t])        # cập nhật SAU khi đã dự báo
    c = lam * c + X[t] * y[t][:, None]
    n = lam * n + 1.0
```

- (1) Không chuẩn hoá → `S` tích luỹ vô hạn, mất cân bằng thang giữa các luồng.
- (2) Ridge hằng số hoặc quá mạnh với SDN (ridge 1e-2 → 5.19 thay vì 4.31) hoặc quá yếu với Géant (529 luồng, nhiều luồng gần 0 → ma trận suy biến, hệ số nổ).
- (3) `anchor` chống trôi tham số ở luồng thưa dữ liệu.

`[KIỂM THỬ]` `tests/test_pfar_causality.py`: đảo giá trị `y[t:]` thành NaN, khẳng định `run()` vẫn trả kết quả hữu hạn cho mọi `t' < t`. Rò rỉ nhân quả sẽ làm test này fail.

### 8.3. Vector ngữ cảnh mở rộng

`features/temporal_features.py::extract_context_features_torch` hiện trả 4 chiều `[σ_local, spike_flag, tod, dow]`. Mở rộng lên 12 chiều:

| Nhóm | Đặc trưng | Trạng thái |
|---|---|---|
| Biến động | `σ_local`, `spike_flag` | đã có |
| Thời gian | `sin/cos(tod)`, `sin/cos(dow)` | thay dạng thô bằng sin/cos |
| **Sai số gần đây từng nhánh** | `EWMA(e²)` cho G, L, M (α = 0.1) | **thêm mới — quan trọng nhất** |
| **Bất đồng giữa nhánh** | `\|ŷ^G−ŷ^M\|`, `\|ŷ^L−ŷ^M\|`, `var(ŷ)` | **thêm mới** |

Nhóm "sai số gần đây" là thiếu sót nghiêm trọng nhất của thiết kế cũ: cổng hoàn toàn không biết nhánh nào vừa dự báo sai. Đây chính là thông tin mà RLS khai thác được.

**Tương thích ngược `[BẮT BUỘC]`:** giữ 4 chiều đầu đúng thứ tự cũ, để `train_gate_stacking.py` (cấu hình ablation A5) vẫn chạy được với `context[..., :4]`.

`EWMA(e²)` chỉ tính từ nhãn đã quan sát ở các bước trước — kiểm tra kỹ tính nhân quả.

### 8.4. `training/train_pfar.py` — tạo mới

Thay thế `train_gate_stacking.py` trong luồng chính.

```python
def select_hyperparams(ds_key: str, P_fit, y_fit) -> dict:
    """Chọn (ridge, lam, anchor) bằng rolling-origin trên Train+Val.
    TUYỆT ĐỐI KHÔNG đụng tập Test (I1).

    Khối chọn siêu tham số = [20% cuối Train | toàn bộ Val]
    Chia thành 4 origin liên tiếp; với mỗi origin: khớp trên phần trước,
    đánh giá trên khối kế tiếp; chọn cấu hình tối thiểu MSE trung bình.

    Lưới dò:
      ridge  ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}
      lam    ∈ {0.99, 0.999, 0.9995, 1.0}
      anchor ∈ {0.0, 0.1, 0.3}
    """

def train_pfar_for_run(ds_key: str, run_id: int, use_oof: bool = True) -> dict:
    """Trả metrics cho cả hai chế độ: pfar_offline, pfar_online."""
```

**Ghi kết quả:** `results/results_PFAR_data_{ds}.csv`, mỗi dòng một run, cột: `run, mode, mse, mae, rmse, skill_persist, ridge, lam, anchor, inference_time_ms`.

**Đo thời gian suy diễn cho đúng `[BẮT BUỘC]`:** báo cáo cũ ghi 0.00–0.02 ms vì chỉ tính tầng gộp. Giá trị mới phải là `t_global + t_local + t_ml + t_pfar`, báo cáo cả p50 và p95.

---

## 9. T6 — Ma trận out-of-fold `[BẮT BUỘC]`

### 9.1. Vì sao

MSE các nhánh trên Validation so với Test:

| | Val | Test | Tỷ lệ |
|---|---|---|---|
| Géant – Global | 3.292 | 0.801 | 4.1× |
| Géant – Local | 4.949 | 0.854 | 5.8× |
| Abilene – ML | 0.961 | 2.763 | 0.35× |

Lát cắt 10% Validation là một chế độ vận hành khác hẳn Test. Khớp bộ gộp trên đó là học sai bài.

### 9.2. `training/build_oof.py` — tạo mới

```
Train (70%)                                    Val(10%)   Test(20%)
├─ fold1 ─┬─ fold2 ─┬─ fold3 ─┬─ fold4 ─┬─ fold5 ─┤
          └─ huấn luyện trên các fold TRƯỚC → dự đoán fold hiện tại
```

```python
def build_oof_matrix(ds_key: str, run_id: int, K: int = 5) -> dict:
    """Blocked expanding-window CV theo trật tự thời gian.
    Fold 1 bị bỏ (không có lịch sử). Phủ ~80% tập Train.

    Trả: {'P_oof': [M_oof, N, 3], 'y_oof': [M_oof, N], 'manifest': ...}
    Ghi ra: cache/{ds}_run_{r}_oof.pt
    """
```

**Chi phí:** fold `k` huấn luyện trên `(k−1)/K` dữ liệu Train. Tổng khối lượng qua 5 fold = `(1+2+3+4)/5 = 2.0×` một lần huấn luyện đầy đủ, cộng mô hình cuối trên toàn Train = **3.0× tổng cộng** (không phải 5×).

**Tối ưu chi phí `[BẮT BUỘC trên Kaggle]`:** chỉ sinh OOF cho **3 seed** (run 0, 1, 2), rồi dùng ma trận OOF của `run_id % 3` cho cả 10 run. Ghi rõ điều này trong phần phương pháp của luận văn. Tiết kiệm ~12 h GPU.

**Hạ `K` từ 5 xuống 3 nếu quota căng:** chi phí giảm còn `(1+2)/3 + 1 = 2.0×`.

---

## 10. T7 — Baseline tầm thường `[BẮT BUỘC — chạy TRƯỚC khi huấn luyện]`

`evaluation/baselines_naive.py`:

```python
def persistence(y_prev): ...        # ŷ_t = x_{t-1}
def historical_average(train_norm, tod, dow): ...   # TB theo (luồng, tod, dow) trên Train
def seasonal_naive(x, period): ...  # ŷ_t = x_{t-T_day}
```

`[SỐ THAM CHIẾU]` Persistence MSE ×10⁻³: SDN **26.552**, Géant **0.855**, Abilene **2.363**.

**Hệ quả phải đưa vào luận văn:** persistence tốt hơn GWN đã công bố trên Géant (0.855 vs 0.879) và tốt hơn 62% trên Abilene (2.363 vs 6.220). Đây là thông tin bắt buộc phải nêu — nếu không, người phản biện sẽ tự phát hiện.

Thêm chỉ số **skill score** vào mọi bảng kết quả:

$$\text{Skill} = 1 - \frac{\text{MSE}_{\text{model}}}{\text{MSE}_{\text{persistence}}}$$

Giá trị hiện tại của mô hình đề xuất: SDN 0.838, Abilene 0.314, Géant 0.046. Con số này cho thấy ngay SDN mới là tập dữ liệu có tín hiệu thật để học.

---

## 11. T8 — Cổng nghiệm thu `[BẮT BUỘC]`

`evaluation/acceptance_gates.py` — chạy tự động sau mỗi run, ghi `logs/<model>/<run>/gates.json`.

| Cổng | Điều kiện | Bắt lỗi |
|---|---|---|
| **C1 — nhánh còn sống** | `mean((ŷ − x_{t−1})²) > 0.05 × MSE_persistence` | B1 |
| **C2 — có học thật** | `train_loss[-1] < 0.95 × train_loss[warmup]` | B1 |
| **C3 — vượt baseline** | `MSE_branch < MSE_persistence` | nhánh vô dụng |
| **C4 — khớp mục tiêu** | `lossfn` là `MSELoss` | B3/I4 |
| **C5 — cache đồng nhất** | mọi cache cùng dataset có `manifest` tương thích | B2 |
| **C6 — không rò rỉ** | tập ES của ML ∩ tập khớp PFAR = ∅ | B4/I5 |

```python
def run_all_gates(ds_key: str, run_id: int) -> dict:
    """Trả {'C1': True/False, ..., 'passed': bool, 'details': {...}}.
    Nếu passed=False -> pipeline phải DỪNG, không được chạy tiếp sang bộ gộp (I6)."""
```

C1 và C2 là hai cổng quan trọng nhất — nếu đã có từ đầu thì lỗi B1 đã bị bắt ngay ngày đầu tiên thay vì phát hiện ở giai đoạn viết luận văn.

---

## 12. T9 — Ablation và kiểm định

### 12.1. Tám cấu hình bắt buộc

Viết lại `evaluation/ablation_study.py`. `ABLATION_CONFIGS` mới:

| # | Mã | Cấu hình | Trả lời |
|---|---|---|---|
| A1 | `persistence` | ŷ = x_{t−1} | Bài toán có tầm thường không? |
| A2 | `best_single` | Nhánh mạnh nhất đơn lẻ | Gộp có cần thiết không? |
| A3 | `best_single_recalib` | Nhánh mạnh nhất + affine theo luồng | **Bao nhiêu phần cải thiện chỉ từ hiệu chỉnh?** |
| A4 | `static_average` | Trung bình cộng 1/3 | Gộp ngây thơ đủ chưa? |
| A5 | `softmax_gate` | Cổng lồi cũ (`train_gate_stacking.py`) | Mô hình cũ đứng ở đâu? |
| A6 | `pfar_offline_val` | PFAR tĩnh, khớp trên Val | Đóng góp của bỏ ràng buộc lồi |
| A7 | `pfar_offline_oof` | PFAR tĩnh, khớp trên OOF | Đóng góp của T6 |
| A8 | `pfar_online` | PFAR trực tuyến | Đóng góp của hiệu chỉnh thích ứng |

**A3 là dòng quan trọng nhất và cũng nguy hiểm nhất.** Số hiện tại trên SDN: A2 = 5.509, A3 = 4.494, A8 = 4.312. Nghĩa là hiệu chỉnh affine một nhánh lấy 18% mức cải thiện, hai nhánh học sâu cộng lại chỉ thêm ~3%.

Phân rã đóng góp đã đo trên SDN (PFAR offline, các tổ hợp nhánh):

| Tổ hợp | MSE ×10⁻³ |
|---|---|
| chỉ ML + hiệu chỉnh | 4.494 |
| Global + Local (không ML) | 14.589 |
| Global + ML | 4.395 |
| Local + ML | 4.351 |
| cả ba | 4.403 |

Agent phải sinh đúng bảng này cho cả ba tập. Không được bỏ A3 khỏi báo cáo.

### 12.2. `evaluation/stats_tests.py` — tạo mới

```python
def wilcoxon_signed_rank(a, b): ...      # thay t-test khi phương sai suy biến
def holm_bonferroni(pvalues): ...        # trả p hiệu chỉnh, đưa vào THẲNG bảng
def diebold_mariano(e1, e2, h=1): ...    # so sánh chuỗi sai số trên Test
```

Lý do thay: `LocalSpatialTCN` có `std = 0.000` qua 10 run nên giả định của t-test ghép cặp bị vi phạm. Diebold–Mariano không phụ thuộc số lần chạy lại nên áp dụng được cả cho persistence và XGBoost (vốn gần như tất định).

Mọi bảng kiểm định phải có cột `p_raw` **và** `p_holm`.

---

## 13. T10 — Điều phối

`run_pipeline_v6.py`:

```bash
# Giai đoạn 0 — xác thực trước, 5 phút, không GPU
python tools/quick_validate_pfar.py

# Giai đoạn 1 — làm sạch
rm -rf cache/ logs/ results/
python run_pipeline_v6.py --stage baselines --dataset all

# Giai đoạn 2 — huấn luyện nhánh (một dataset mỗi session Kaggle)
python run_pipeline_v6.py --stage branches --dataset sdn     --runs 10 --skip_existing
python run_pipeline_v6.py --stage branches --dataset geant   --runs 10 --skip_existing
python run_pipeline_v6.py --stage branches --dataset abilene --runs 10 --skip_existing

# Giai đoạn 3 — OOF (3 seed)
python run_pipeline_v6.py --stage oof --dataset all --oof_seeds 3 --folds 5

# Giai đoạn 4 — cache + PFAR + ablation + kiểm định (CPU, nhanh)
python run_pipeline_v6.py --stage combine --dataset all --runs 10
python run_pipeline_v6.py --stage report  --dataset all
```

`--skip_existing` là bắt buộc trên Kaggle: session bị ngắt ở giờ thứ 11 là chuyện thường. Trước khi bỏ qua một run, phải đối chiếu manifest — run cũ có manifest lệch thì chạy lại chứ không skip.

---

## 14. Ngân sách thời gian GPU Kaggle

Quy đổi từ thời gian suy diễn đo được trong `logs/` (STWaveFormer 3.2–3.6 ms/batch-64, LocalTCN 1.5–1.7 ms/batch-64), giả định bước huấn luyện ≈ 3× bước suy diễn, số epoch theo log thực tế, hệ số 2.5 cho overhead dataloader và GPU Kaggle.

| Giai đoạn | Nội dung | Thời gian |
|---|---|---|
| T0 | Xác thực PFAR trên cache sẵn có | ~5 phút CPU |
| T7 | Baseline tầm thường | ~10 phút CPU |
| T3 | 2 nhánh DL × 3 tập × 10 run | **~8 h** |
| T4 | XGBoost × 3 tập × 10 run, `device="cuda"` | **~1,5 h** |
| T2/T5 | Precompute cache + PFAR + kiểm định | ~0,5 h |
| T6 | OOF, K=5, 3 seed | **~6 h** |
| | **Tổng** | **~16 h** |

Nằm gọn trong quota 30 h/tuần. Abilene chiếm quá nửa chi phí (33.6k cửa sổ huấn luyện, gấp 8 lần SDN) — nếu ép tiến độ thì chạy SDN + Géant trước.

**Hiệu chỉnh ước lượng trước khi chạy full:** chạy đúng một run `STWaveFormer` trên Abilene, đo thời gian mỗi epoch, nhân lên. Sai số của bảng trên có thể tới ±50% vì suy ra từ thời gian suy diễn chứ không đo trực tiếp.

---

## 15. Kết quả kỳ vọng

MSE ×10⁻³ trên Test:

| Tập | Mô hình cũ | PFAR (đã đo, chưa sửa nhánh) | Sau khi sửa nhánh (kỳ vọng) |
|---|---|---|---|
| SDN | 6.562 ± 0.211 | **4.312 ± 0.042** | 3,9 – 4,3 |
| Géant | 0.796 ± 0.025 | 0.816 ± 0.005 | 0,74 – 0,80 |
| Abilene | 2.731 ± 0.866 | **1.620 ± 0.002** | 1,5 – 1,62 |

**Trần lý thuyết cần biết trước:** oracle chọn nhánh tốt nhất cho từng mẫu trên SDN là **3.597×10⁻³**. Không cơ chế gộp nào vượt được ngưỡng đó với ba nhánh hiện tại. Khoảng từ 4.31 xuống 3.60 là **toàn bộ** dư địa còn lại của hướng cải tiến bộ gộp (~16%). Muốn đi xa hơn phải cải thiện bản thân các nhánh.

**Trung thực về Géant:** PFAR hơi kém hơn nhánh Global đơn lẻ. Không được che giấu. Lý do đúng là ở tầm dự báo một bước, Géant gần như là bài toán persistence (skill score chỉ 0.046), nên không cơ chế gộp nào tạo ra khác biệt có ý nghĩa.

---

## 16. Những việc agent KHÔNG được làm

| # | Cấm | Vì sao |
|---|---|---|
| N1 | Tinh chỉnh thêm cổng softmax lồi | Trần đã chứng minh bằng quy hoạch toàn phương: chặn ở đúng mức XGBoost trên SDN |
| N2 | Thêm nhánh dự báo thứ tư | Phải để ba nhánh hiện tại qua C1–C4 trước |
| N3 | Tăng dung lượng ST-WaveFormer trên SDN | Vấn đề là số mẫu hiệu dụng, không phải dung lượng; mô hình lớn hơn sẽ tệ hơn |
| N4 | Chọn bất kỳ siêu tham số nào trên Test | Vi phạm I1; ridge lệch ba bậc độ lớn giữa các tập nên đây là chỗ dễ trượt nhất |
| N5 | Cắt/đệm mảng để "khớp kích thước" | Vi phạm I2; chính là nguyên nhân B2 |
| N6 | Bỏ qua cổng nghiệm thu khi thấy kết quả "vẫn ổn" | Vi phạm I6 |
| N7 | Xoá `training/train_gate_stacking.py` | Cần cho ablation A5 |
| N8 | Chạy lại GWN / BiGRU | Ngoài phạm vi; giữ số trích bài báo gốc làm tham chiếu |

---

## 17. Checklist bàn giao

- [ ] `tools/quick_validate_pfar.py` chạy PASS mọi ô `[SỐ THAM CHIẾU]`
- [ ] `manifest.json` tồn tại trong mọi thư mục run; `assert_compatible` không raise ở bất kỳ đâu
- [ ] Mọi nhánh, mọi run, mọi dataset: `gates.json` có `passed: true`
- [ ] `mean_abs_delta` của LocalSpatialTCN trên Géant và Abilene **> 0** ở epoch cuối (chứng minh B1 đã sửa)
- [ ] Bảng baseline tầm thường có mặt trong mọi bảng kết quả, kèm cột skill score
- [ ] `results/ablation_v6.csv` đủ 8 cấu hình × 3 tập × 10 run
- [ ] Bảng kiểm định có cả `p_raw` và `p_holm`; cặp có phương sai suy biến dùng Wilcoxon
- [ ] Thời gian suy diễn báo cáo là tổng 3 nhánh + PFAR, có p50 và p95
- [ ] `tests/` pass: `test_manifest.py`, `test_pfar_causality.py`, `test_no_leakage.py`
- [ ] README ghi rõ ridge/λ/anchor được chọn bằng rolling-origin trên Train+Val, kèm giá trị cuối cho từng tập
