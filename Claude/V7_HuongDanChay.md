# v7 — Hướng dẫn chạy & những gì đã thay đổi

Tài liệu này là bản thực thi của `QuyTrinh_ThucNghiem_Moi_v6.md` sau khi đã kiểm
chứng lại toàn bộ trên chính cache trong repo, kèm 5 sửa đổi (S1–S5).

---

## 1. Kiểm chứng ngược v6 — cái gì đúng

Chạy lại độc lập trên 60 file cache + dữ liệu gốc, không lấy số từ tài liệu:

| Kiểm chứng | v6 công bố | Đo lại |
|---|---|---|
| Abilene Global run 0–4 / 5–9 | 10.38 / 1.98 | **10.37 / 1.98** ✅ lỗi B2 có thật |
| Géant Global run 0–4 / 5–9 | 0.864 / 0.738 | **0.864 / 0.738** ✅ |
| Persistence SDN/Géant/Abilene | 26.552 / 0.855 / 2.363 | **26.554 / 0.855 / 2.363** ✅ |
| Oracle tổ hợp lồi SDN | 5.509, `w=[0,0,1]` | **5.509, `w=[0,0,1]`** ✅ |
| PFAR offline / online SDN | 4.403 / 4.312 | **4.402 / 4.312** ✅ |

Kết luận cốt lõi của v6 đứng vững: **trên SDN, tổ hợp lồi tốt nhất trong vũ trụ là
`w = [0, 0, 1]`**. Cổng softmax (`Graph_models/contextual_gate.py`) không thể thắng
XGBoost, chỉ có thể hoà. Phải bỏ ràng buộc lồi.

## 2. Năm sửa đổi so với v6

### S1 — Bỏ tuyên bố "oracle 3.597 là trần của mọi cơ chế gộp"

v6 (Mục 11, Mục 15, N1) khẳng định không cơ chế gộp nào vượt được `min_k(p_k − y)²`.
**Sai.** Đó là trần của việc *chọn cứng một nhánh*, không phải của hồi quy. Phản ví
dụ: `p₁ = y+e`, `p₂ = y−e` → trung bình cho sai số 0 < `min = e²`.
Test `test_convex_bi_chan_boi_nhanh_manh_nhat` trong `tests/test_v7_invariants.py`
kiểm chứng điều này. Hệ quả: v6 tự khoá dư địa của chính nó.

### S2 — Bộ gộp hai tầng (`Graph_models/stacker.py`)

v6 giới hạn ở affine. Đo trên cache hiện có (khớp Val, đánh giá Test, không rò rỉ):

| SDN | MSE ×10⁻³ |
|---|---|
| XGBoost đơn lẻ | 5.509 |
| ST-Adaptive-Ensemble (softmax lồi) | 6.562 |
| PFAR online (v6) | 4.312 |
| **Hai tầng, tĩnh** | **4.146** |
| **Hai tầng, trực tuyến** | **4.088** |

Nhưng trên Géant/Abilene, tầng phi tuyến khớp trên Val lại **tệ hẳn** (Géant 1.229
so với 0.815) vì Val là một lát cắt 10% của một chế độ duy nhất. Vì vậy có **cổng
C7**: tầng phi tuyến phải cải thiện ≥1% trên khối kiểm định, nếu không thì tự động
rút về PFAR affine thuần. Không ép một bộ gộp cho cả ba tập.

### S3 — OOF đủ 10 seed, không dùng chung giữa các run

SPEC §9.2 đề nghị sinh OOF cho 3 seed rồi dùng `run_id % 3` cho cả 10 run. Với
stacker phi tuyến, điều đó tạo tương quan giả giữa các run và làm `std` 10 run mất
ý nghĩa — tái hiện lỗi B2 ở dạng nhẹ. Thiếu quota thì hạ `K` từ 5 xuống 3
(chi phí 2.0×), **không** dùng chung OOF.

### S4 — Hồi sinh nhánh DL trên SDN là ưu tiên cao, không phải tuỳ chọn

Dưới bộ gộp phi tuyến, đóng góp của hai nhánh học sâu trên SDN:

```
chỉ ML 4.181  →  +Global 4.088  →  +Local 4.087  →  cả ba 4.076
```

**2.5%.** Đây là dòng phản biện sẽ hỏi. v6 đề nghị biện hộ bằng "tính bền vững
xuyên tập dữ liệu" — đó là biện hộ, không phải sửa chữa.

### S5 — Quy tắc 1-SE khi chọn siêu tham số *(phát sinh trong quá trình cài đặt)*

Đây là vấn đề v6 **chưa từng kiểm tra**. SPEC §7.3 yêu cầu chọn ridge bằng
rolling-origin trên Train+Val. Khi cài đặt và đo thật:

| ridge | MSE rolling-origin (Val) — Géant | MSE Test — Géant |
|---|---|---|
| 1e-5 | 0.928 | 2.483 |
| 1e-4 | **0.922** ← argmin | 1.502 |
| 1e-3 | 0.925 | 0.895 |
| 1e-2 | 0.977 | **0.814** |
| 1e-1 | 1.197 | 1.056 |

Tín hiệu chọn từ Val **phẳng** (biến thiên 6%) trong khi Test biến thiên **300%**.
Ridge đơn giản không định danh được từ Val, và argmin chỉ đang bắt nhiễu.

Quy tắc 1-SE (chọn ridge lớn nhất trong phạm vi 1 sai số chuẩn của argmin, với
`lam`/`anchor` giữ ở cấu hình argmin) cải thiện cả ba tập mà **không đụng Test**.

**Hệ quả phải ghi vào luận văn:** con số Géant 0.816 của v6 chỉ đạt được khi dò
ridge trên Test. Số trung thực với quy trình chọn hợp lệ:

| | v6 công bố (dò trên Test) | **v7 trung thực (chọn trên Val)** |
|---|---|---|
| SDN | 4.312 | **4.393** |
| Géant | 0.816 | **1.153** |
| Abilene | 1.620 | **1.663** |

Trên Géant, 1.153 **kém hơn** nhánh Global đơn lẻ (0.738). Phải nói thẳng điều đó.
Đây cũng là lý do OOF (S3) cần thiết không chỉ để khớp tham số mà còn để *chọn*
tham số.

---

## 3. Các file đã thêm / sửa

**Thêm mới**

| File | Vai trò |
|---|---|
| `features/manifest.py` | Khoá dữ liệu, chống lỗi B2 tái diễn (I3) |
| `Graph_models/pfar.py` | PFAROffline / PFAROnline (RLS có quên) |
| `Graph_models/stacker.py` | Bộ gộp hai tầng + cổng C7 (S2) |
| `training/train_pfar.py` | Chọn siêu tham số rolling-origin + quy tắc 1-SE (S5) |
| `training/build_oof.py` | Ma trận out-of-fold (Stage B) |
| `evaluation/baselines_naive.py` | Persistence, HA, seasonal naive, skill score |
| `evaluation/acceptance_gates.py` | Cổng C1–C7 |
| `evaluation/stats_tests.py` | Wilcoxon, Holm–Bonferroni, Diebold–Mariano |
| `evaluation/ablation_v7.py` | Ablation 10 cấu hình (A1–A10) |
| `run_pipeline_v7.py` | Điều phối toàn bộ |
| `tools/quick_validate_v7.py` | Xác thực trên cache sẵn có, không GPU |
| `tests/test_v7_invariants.py` | 6 test bất biến, gồm test rò rỉ nhân quả |

**Đã sửa**

| File | Thay đổi |
|---|---|
| `run_experiments.py` | `MSELoss` thay `SmoothL1Loss(beta=0.01)` (I4); warmup khoá early stopping 15 epoch; nhóm tham số riêng cho `out_head` (lr×10, weight_decay=0); log `mean_abs_delta`; ghi manifest |
| `Graph_models/local_filters.py` | `forward(..., return_delta=True)` |
| `baselines_ml/xgboost_baseline.py` | `fit(X_train, y_train, X_es, y_es)` + `split_train_es()` (I5); hỗ trợ `device="cuda"` |
| `training/precompute_cache.py` | **Xoá khối reuse `y_pred_data.npy`** (nguồn gốc B2); checkpoint bắt buộc; đối chiếu manifest; lưu `x_last` + `manifest` vào cache; ML early-stop trên Train-ES |
| `features/feature_store.py` | Sinh manifest, thêm tham số `seed` |

**Giữ nguyên:** `training/train_gate_stacking.py` (cần cho ablation A5).

---

## 4. Chạy

```bash
# Giai đoạn 0 — xác thực, ~10 phút, KHÔNG cần GPU
python tools/quick_validate_v7.py --datasets sdn
python -m pytest tests/ -q

# Giai đoạn 1 — làm sạch + baseline tầm thường
python run_pipeline_v7.py --stage clean
python run_pipeline_v7.py --stage baselines --dataset all

# Giai đoạn 2 — Stage A: huấn luyện nhánh (một dataset mỗi session Kaggle)
python run_pipeline_v7.py --stage branches --dataset sdn     --runs 10
python run_pipeline_v7.py --stage branches --dataset geant   --runs 10
python run_pipeline_v7.py --stage branches --dataset abilene --runs 10

# Giai đoạn 3 — cache + bộ gộp + ablation (CPU, nhanh)
python training/precompute_cache.py --datasets all --runs 10
python run_pipeline_v7.py --stage combine --dataset all --runs 10

# Giai đoạn 4 — Stage B: OOF rồi chạy lại combine
python run_pipeline_v7.py --stage oof --dataset all --runs 10 --folds 5
python run_pipeline_v7.py --stage combine --dataset all --runs 10
```

`--stage clean` **đổi tên** `cache/ logs/ results/` thành `*_v6_backup/` chứ không
xoá, để còn đối chiếu khi viết luận văn.

## 5. Kiểm tra bắt buộc sau khi training xong

- [ ] `mean_abs_delta` của LocalSpatialTCN trên Géant và Abilene **> 0** ở epoch
      cuối — chứng minh lỗi B1 đã sửa.
- [ ] Mọi `logs/*/run_*/gates.json` có `passed: true`.
- [ ] Mọi cache cùng dataset có `n_windows_test` giống nhau (cổng C5).
- [ ] Bảng ablation đủ 10 dòng × 3 tập; **A3 và A10 phải có mặt trong luận văn**.
- [ ] Báo cáo song song chế độ tĩnh và trực tuyến. Chế độ trực tuyến dùng nhãn thật
      trên Test để hiệu chỉnh — hợp lệ trong giám sát lưu lượng, nhưng phải nêu rõ
      giả định ở Mục 3. Chỉ khoe 1.620 mà giấu 2.280 tĩnh sẽ bị coi là rò rỉ.
- [ ] Ghi rõ ridge/λ/anchor được chọn bằng rolling-origin 1-SE trên Val (hoặc OOF),
      kèm giá trị cuối cho từng tập.
