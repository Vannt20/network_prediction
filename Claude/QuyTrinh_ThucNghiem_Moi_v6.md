# QUY TRÌNH THỰC NGHIỆM MỚI — ST-Adaptive-Ensemble v6
### Đặc tả đầy đủ, kèm cổng nghiệm thu và số liệu kiểm chứng

---

## 0. "Tốt nhất" ở đây nghĩa là gì

Không ai đảm bảo được một quy trình là tối ưu tuyệt đối, và bất kỳ ai hứa điều đó đều đang nói quá. Vì vậy tôi định nghĩa rõ tiêu chuẩn mà quy trình này được thiết kế để đạt, và bạn có thể kiểm tra từng tiêu chuẩn một:

| # | Tiêu chuẩn | Cách kiểm tra |
|---|---|---|
| T1 | **Không rò rỉ thông tin** ở bất kỳ mắt xích nào | Mọi tham số chỉ khớp trên dữ liệu trước mốc dự báo; có cổng kiểm tra tự động |
| T2 | **Tái lập được** — chạy lại cho cùng số | Manifest hash dữ liệu + seed + commit cho mỗi run |
| T3 | **Mọi thành phần đều phải tự chứng minh** | Ablation bắt buộc; thành phần nào không đóng góp thì loại bỏ, không giữ cho đẹp sơ đồ |
| T4 | **Vượt được baseline tầm thường** | So với persistence và HA, không chỉ so với GWN trong bài báo |
| T5 | **Biết trước trần lý thuyết** | Tính oracle trước khi tối ưu, để không đốt thời gian đuổi theo thứ không tồn tại |

Quy trình dưới đây đã được kiểm chứng ngược trên chính 60 file cache trong repo của bạn: các con số ở Mục 9 là số đo thật, không phải số kỳ vọng.

---

## 1. Kết quả kiểm chứng — lý do quy trình phải thay đổi

Tôi đã cài đặt bộ gộp mới và chạy nó trên đúng dự đoán ba nhánh hiện có của bạn, với giao thức nhân quả tuyệt đối (mỗi bước thời gian chỉ dùng thông tin đến bước trước đó).

**MSE ×10⁻³ trên tập Test:**

| | SDN | Géant | Abilene¹ |
|---|---|---|---|
| Persistence (baseline tầm thường) | 26.552 | 0.855 | 2.363 |
| GWN (bài báo gốc) | 7.936 | 0.879 | 6.220 |
| ST-WaveFormer (Global) | 15.350 | 0.801 | 1.978 |
| LocalSpatialTCN (Local) | 15.152 | 0.854² | 2.361² |
| XGBoost (ML) | 5.509 | 0.915 | 2.763 |
| **ST-Adaptive-Ensemble hiện tại (softmax lồi)** | **6.562 ± 0.211** | **0.796 ± 0.025** | **2.731 ± 0.866** |
| Trần lý thuyết của mọi tổ hợp lồi | 5.509 | 0.735 | — |
| Gộp affine theo luồng, khớp tĩnh trên Val | 4.403 ± 0.132 | 0.794 | 2.314 |
| **Gộp affine theo luồng + hiệu chỉnh trực tuyến (đề xuất)** | **4.312 ± 0.042** | **0.816 ± 0.005** | **1.620 ± 0.002** |

¹ Abilene tính trên run 5–9; run 0–4 chứa artifact hỏng đã nêu trong báo cáo phân tích.
² Hai giá trị này thực chất là persistence — nhánh Local chưa học được gì trên hai tập đó.

**Đọc bảng này:**

- **SDN: 6.562 → 4.312**, giảm 34.3%; và lần đầu tiên **vượt XGBoost 21.7%**. Đây chính là điều bạn đang cần.
- **Abilene: 2.731 → 1.620**, giảm 40.7%; vượt nhánh mạnh nhất 18.1% và vượt persistence 31.4%. Bài toán "cổng không ổn định trên Abilene" biến mất.
- **Độ lệch chuẩn sụp đổ**: SDN từ ±0.211 xuống ±0.042; Abilene từ ±0.866 xuống ±0.002. Luận điểm "cơ chế gộp là bộ điều hoà phương sai" của bạn giờ mới thực sự có bằng chứng.
- **Géant: 0.796 → 0.816**, hơi kém đi. Cần trung thực: trên Géant không cơ chế nào vượt được nhánh Global đơn lẻ một cách có ý nghĩa, vì bản thân Géant ở tầm dự báo một bước gần như là bài toán persistence. Quy trình mới sẽ xử lý bằng cách báo cáo thẳng điều này thay vì giấu.

---

## 2. Sơ đồ quy trình mới

```
G0  KHOÁ DỮ LIỆU        ──►  manifest.json (hash, split, seed, commit)
                              │
G1  BASELINE TẦM THƯỜNG ──►  Persistence, HA, ARIMA nhẹ
                              │
G2  HUẤN LUYỆN 3 NHÁNH  ──►  Global / Local / ML   ◄── cổng nghiệm thu C1–C4
                              │
G3  SINH MA TRẬN OOF    ──►  blocked CV 5 fold trên Train
                              │            + Val giữ nguyên cho early stopping
                              ▼
G4  BỘ GỘP PFAR         ──►  Per-Flow Adaptive Affine Recalibrator
                              │   • khớp offline trên OOF
                              │   • hiệu chỉnh trực tuyến bằng RLS có quên
                              ▼
G5  ĐÁNH GIÁ KÉP        ──►  chế độ tĩnh  +  chế độ trực tuyến
                              │
G6  ABLATION BẮT BUỘC   ──►  8 cấu hình
                              │
G7  KIỂM ĐỊNH           ──►  Wilcoxon + Holm–Bonferroni + Diebold–Mariano
```

Điểm khác biệt cốt lõi so với quy trình cũ nằm ở **G3 và G4**. Quy trình cũ huấn luyện cổng trên một lát cắt 10% Validation và ép trọng số nằm trên đơn hình. Quy trình mới huấn luyện bộ gộp trên toàn bộ Train qua dự đoán out-of-fold, bỏ ràng buộc lồi, và cho phép bộ gộp tự hiệu chỉnh theo thời gian.

---

## 3. G0 — Khoá dữ liệu và manifest

Mọi lần chạy phải sinh `manifest.json`:

```json
{
  "dataset": "abilene",
  "csv_sha256": "…",
  "n_steps_after_hygiene": 48096,
  "split": {"train": 33667, "val": 4810, "test": 9619},
  "seq_len": 24, "k_lags": 12,
  "n_windows": {"val": 4786, "test": 9595},
  "seed": 42, "git_commit": "…",
  "pipeline_version": "v6"
}
```

**Quy tắc cứng:** mọi bước tiêu thụ artifact phải đối chiếu `manifest` và dừng nếu lệch. Không bao giờ cắt mảng ngầm. Đây chính là lỗi đã khiến 5/10 run Abilene và Géant của bạn dùng artifact từ pipeline cũ.

```python
def assert_compatible(manifest, artifact_manifest):
    keys = ['csv_sha256', 'split', 'seq_len', 'pipeline_version']
    for k in keys:
        assert manifest[k] == artifact_manifest[k], \
            f"Artifact không tương thích ở trường '{k}'. Hãy xoá logs/cache và chạy lại."
```

---

## 4. G1 — Baseline tầm thường (chạy trước, không phải sau)

Ba baseline bắt buộc, chạy **trước khi** huấn luyện bất cứ mô hình nào:

| Baseline | Định nghĩa |
|---|---|
| Persistence | `ŷ_t = x_{t−1}` |
| Historical Average | trung bình theo (luồng, time-of-day, day-of-week) trên Train |
| Seasonal naive | `ŷ_t = x_{t−T_day}` |

Lý do đặt ở G1 chứ không phải G6: nếu một nhánh học sâu không vượt được persistence sau khi huấn luyện, đó là tín hiệu dừng ngay lập tức, không phải phát hiện ở cuối luận văn. Với Géant và Abilene, nếu bạn chạy bước này từ đầu thì đã phát hiện nhánh Local chết ngay ngày đầu tiên.

Thêm một chỉ số vào mọi bảng kết quả:

$$\text{Skill}_{\text{persist}} = 1 - \frac{\text{MSE}_{\text{model}}}{\text{MSE}_{\text{persistence}}}$$

Đây mới là con số nói lên mô hình học được gì. Với số liệu hiện tại: SDN 0.838, Abilene 0.314, Géant 0.046. Nó cho thấy ngay rằng SDN mới là tập dữ liệu có câu chuyện khoa học, còn Géant thì gần như không.

---

## 5. G2 — Huấn luyện ba nhánh, kèm bốn cổng nghiệm thu

Mỗi nhánh chỉ được coi là hợp lệ khi qua đủ bốn cổng. Cài đặt thành hàm kiểm tra chạy tự động sau mỗi run.

| Cổng | Điều kiện | Bắt lỗi gì |
|---|---|---|
| **C1 — Nhánh còn sống** | `mean(\|ŷ_branch − x_{t−1}\|²) > 0.05 × MSE_persistence` | Nhánh sụp về persistence (lỗi của Local hiện tại) |
| **C2 — Có học thật** | `train_loss` cuối < 0.95 × `train_loss` epoch 1 | Early stopping giữ lại đúng trạng thái khởi tạo |
| **C3 — Vượt baseline** | `MSE_branch < MSE_persistence` | Nhánh vô dụng |
| **C4 — Khớp mục tiêu** | hàm mất mát cùng họ với chỉ số công bố | Huấn luyện bằng Huber δ=0.01 rồi công bố MSE |

Các sửa đổi cụ thể cho G2:

**Nhánh Local** (`Graph_models/local_filters.py`):
- Giữ zero-init ở `out_head` nhưng **khoá early stopping trong 15 epoch đầu**, để mô hình có cơ hội thoát khỏi điểm khởi tạo persistence trước khi bị chấm điểm.
- Loại `out_head` khỏi `weight_decay` (weight decay đang kéo tầng zero-init về lại 0).
- Đặt learning rate riêng cho `out_head` cao gấp 10 lần phần còn lại.
- Ghi log `mean(|delta_norm|)` mỗi epoch — đây là chỉ báo sống/chết của nhánh.

**Nhánh Global** (`run_experiments.py`, dòng 288):
- `SmoothL1Loss(beta=0.01)` → `MSELoss()`, hoặc `beta` tăng lên 0.05–0.1 nếu muốn giữ tính bền với ngoại lai. Hiện gần như mọi mẫu rơi vào vùng tuyến tính, tức mạng tối ưu MAE trong khi bạn công bố MSE.
- Thử thêm cấu hình **channel-independent** (mỗi luồng OD là một mẫu, chia sẻ trọng số) — trên SDN chỉ có 4,320 cửa sổ trượt trong khi XGBoost dạng shared model nhìn thấy 846 nghìn dòng. Chênh lệch số mẫu hiệu dụng gần 200 lần là giả thuyết hàng đầu giải thích khoảng cách 15.2 so với 5.5.

**Nhánh ML:**
- Tách tập early stopping ra khỏi tập huấn luyện bộ gộp. Hiện XGBoost early-stop trên chính Validation mà cổng dùng để học trọng số, khiến nhánh ML "đẹp giả" đúng ở nơi ra quyết định. Trên Abilene, hậu quả là bộ gộp dồn 79% trọng số cho nhánh tệ nhất trên Test.
- Với quy trình OOF ở G3, vấn đề này tự biến mất.

---

## 6. G3 — Sinh ma trận dự đoán out-of-fold

Đây là thay đổi nguyên lý quan trọng nhất.

```
Train (70%)                                    Val(10%)   Test(20%)
├─ fold1 ─┬─ fold2 ─┬─ fold3 ─┬─ fold4 ─┬─ fold5 ─┤
          │         │         │         │         │
  huấn luyện trên phần trước → dự đoán fold hiện tại (out-of-fold)
```

Thuật toán:

```
for k in 1..K:                       # K = 5, blocked, theo trật tự thời gian
    train_k = các fold < k
    huấn luyện Global_k, Local_k, ML_k trên train_k
    P_oof[fold_k] = dự đoán của 3 nhánh trên fold_k     # chưa từng thấy fold_k
huấn luyện 3 nhánh cuối cùng trên toàn bộ Train  → dùng để suy diễn Val và Test
```

Thu được `P_oof` phủ ~80% tập Train (bỏ fold 1 vì không có lịch sử). So với quy trình cũ:

| | Quy trình cũ | Quy trình mới |
|---|---|---|
| Dữ liệu huấn luyện bộ gộp | 10% (một lát cắt thời gian duy nhất) | ~56% tổng dữ liệu, phủ nhiều chế độ vận hành |
| Số chế độ vận hành nhìn thấy | 1 | 4–5 |
| Nhánh ML có bị ưu ái không | Có (early stopping cùng tập) | Không |

Lý do bắt buộc: trên Géant, MSE của các nhánh trên Validation cao gấp 3–6 lần so với Test. Huấn luyện bộ gộp trên lát cắt đó là học một chế độ không hề xuất hiện lúc đánh giá. Phủ toàn Train là cách duy nhất để bộ gộp thấy đủ sự đa dạng.

Chi phí: huấn luyện ba nhánh 5 lần thay vì 1. Với SDN (6.2k bước) là không đáng kể; với Abilene (48k bước) mất thêm vài giờ GPU. Đây là chi phí xứng đáng.

---

## 7. G4 — Bộ gộp PFAR (Per-Flow Adaptive Affine Recalibrator)

### 7.1. Dạng hàm

Thay tổ hợp lồi bằng tổ hợp affine riêng cho từng luồng OD:

$$\hat{y}_{t,f} = \alpha_f(t) + \beta^G_f(t)\,\hat{y}^G_{t,f} + \beta^L_f(t)\,\hat{y}^L_{t,f} + \beta^M_f(t)\,\hat{y}^M_{t,f}$$

Ba khác biệt so với công thức (12) trong báo cáo cũ:

1. **Bỏ ràng buộc** `β ≥ 0` và `Σβ = 1`. Ràng buộc lồi chính là thứ chặn trên mô hình tại đúng mức của nhánh mạnh nhất: trên SDN, tổ hợp lồi tốt nhất trong vũ trụ là `w = [0, 0, 1]` cho MSE 5.509 — tức hoà với XGBoost, không bao giờ thắng.
2. **Thêm hệ số chệch** `α_f`. Riêng việc hiệu chỉnh chệch và tỷ lệ cho từng luồng đã đưa XGBoost từ 5.509 xuống 4.494 trên SDN, tức 18% mức cải thiện đến từ đây.
3. **Tham số phụ thuộc thời gian** `β_f(t)`, cập nhật trực tuyến.

### 7.2. Hai chế độ vận hành

**Chế độ A — tĩnh (offline).** Khớp `(α_f, β_f)` bằng ridge trên `P_oof`, cố định trong suốt tập Test. Dùng cho kịch bản triển khai không có phản hồi đo lường.

**Chế độ B — trực tuyến (adaptive).** Khởi tạo từ nghiệm offline, sau đó cập nhật bằng bình phương tối thiểu đệ quy có hệ số quên:

```python
S = λ·S + x_t x_tᵀ        # x_t = [ŷ^G, ŷ^L, ŷ^M, 1] cho từng luồng
c = λ·c + x_t · y_t
n = λ·n + 1
A = S/n + δ·(tr(A)/D)·I    # ridge tỷ lệ với vết, chống suy biến
β = 0.9 · solve(A, c/n) + 0.1 · β₀   # neo mềm về nghiệm offline
```

Ba chi tiết kỹ thuật **bắt buộc**, tôi đã kiểm chứng rằng thiếu chúng thì bộ gộp phân kỳ:

- **Chuẩn hoá theo số mẫu hiệu dụng `n`** thay vì tích luỹ thô. Không có bước này, Géant nổ lên 7.17×10⁻³ (so với 0.816 khi có).
- **Ridge tỷ lệ với vết ma trận**, không phải hằng số. Géant có 529 luồng, nhiều luồng gần như bằng 0 khiến ma trận suy biến; ridge cố định thì hoặc quá mạnh với SDN hoặc quá yếu với Géant.
- **Neo mềm về nghiệm offline** (hệ số 0.1). Chống trôi tham số ở các luồng ít dữ liệu.

### 7.3. Chọn siêu tham số — chỗ dễ rò rỉ nhất

Hệ số ridge `δ` khác nhau ba bậc độ lớn giữa các tập: SDN cần 1e-5, Géant cần 1e-2, Abilene cần 1e-3. Đây là siêu tham số thật, **không được chọn trên tập Test**.

Quy tắc: chọn `δ` và `λ` bằng **rolling-origin trên phần cuối tập Train + Val**, không bao giờ đụng Test.

```
Khối dùng để chọn siêu tham số:  [20% cuối của Train | toàn bộ Val]
  origin 1: khớp trên phần trước → đánh giá 1 khối
  origin 2: mở rộng → đánh giá khối kế
  ...
  chọn (δ, λ) tối thiểu MSE trung bình qua các origin
```

Trong báo cáo, phải ghi rõ `δ` và `λ` được chọn thế nào. Nếu không, người phản biện có quyền giả định bạn đã dò trên Test — và với khoảng cách ba bậc độ lớn giữa các tập thì nghi ngờ đó rất hợp lý.

### 7.4. Giữ lại phần "contextual" của ý tưởng gốc

Đóng góp khoa học mà bạn đã đặt tên — Per-Flow Contextual Meta-Gating — không bị vứt đi, mà được đặt đúng chỗ. Mạng ngữ cảnh giờ sinh ra **hệ số affine** thay vì trọng số softmax, và vector ngữ cảnh được mở rộng từ 4 chiều lên khoảng 12 chiều:

| Nhóm | Đặc trưng | Hiện có? |
|---|---|---|
| Biến động | `σ_local`, `spike_flag` | Có |
| Thời gian | `tod`, `dow` (dạng sin/cos) | Có |
| **Sai số gần đây từng nhánh** | `EWMA(e²)` cho Global, Local, ML | **Thiếu — quan trọng nhất** |
| **Bất đồng giữa các nhánh** | `\|ŷ^G−ŷ^M\|`, `\|ŷ^L−ŷ^M\|`, `var(ŷ)` | **Thiếu** |
| **Danh tính luồng** | `nn.Embedding(N, 8)` | **Thiếu** |

Nhóm "sai số gần đây" là thiếu sót nghiêm trọng nhất của thiết kế cũ: cổng hoàn toàn không biết nhánh nào vừa dự báo sai. Đây chính là thông tin mà RLS khai thác được và là lý do nó thắng.

---

## 8. G5 — Giao thức đánh giá kép

Báo cáo phải trình bày song song hai kịch bản, không chọn một:

| Kịch bản | Giả định triển khai | Bộ gộp |
|---|---|---|
| **Tĩnh** | Không có phản hồi đo lường sau dự báo | PFAR chế độ A |
| **Trực tuyến** | Lưu lượng thực tế đo được sau mỗi chu kỳ, phản hồi về hệ thống | PFAR chế độ B |

Kịch bản trực tuyến là hợp lệ và thực tế trong bài toán giám sát lưu lượng mạng: hệ thống đo được lưu lượng thật tại bước `t` ngay sau khi nó xảy ra, nên hoàn toàn có thể dùng nó để hiệu chỉnh cho bước `t+1`. Nhưng phải **nêu rõ giả định này trong Mục 3** và thảo luận trường hợp độ trễ thu thập lớn hơn một chu kỳ. Không nêu ra thì bị coi là rò rỉ; nêu ra rõ ràng thì đó là một đóng góp về mặt triển khai.

Ba chỉ số bổ sung bắt buộc:

- Skill score so với persistence (Mục 4).
- Sai số phân tầng theo phân vị lưu lượng (10% luồng lớn nhất / phần còn lại) — mạng thật quan tâm luồng lớn.
- Sai số trên các đoạn có trôi dạt phân phối, phát hiện bằng kiểm định Kolmogorov–Smirnov trượt giữa Train và từng cửa sổ Test. Đây chính là bằng chứng cho giả thuyết mà Mục 6 báo cáo cũ đã nêu nhưng chưa chứng minh.

---

## 9. G6 — Ablation bắt buộc (8 cấu hình)

Bảng này thay thế phần Ablation còn thiếu số liệu trong báo cáo cũ. Chạy đủ trên cả ba tập.

| # | Cấu hình | Trả lời câu hỏi |
|---|---|---|
| A1 | Persistence | Bài toán có tầm thường không? |
| A2 | Chỉ nhánh mạnh nhất | Gộp có cần thiết không? |
| A3 | Nhánh mạnh nhất + hiệu chỉnh affine theo luồng | **Bao nhiêu phần cải thiện chỉ đến từ hiệu chỉnh?** |
| A4 | Gộp trung bình cộng tĩnh 1/3 | Gộp ngây thơ đã đủ chưa? |
| A5 | Cổng softmax lồi (mô hình cũ) | Mô hình cũ đứng ở đâu? |
| A6 | PFAR tĩnh, khớp trên Val | Đóng góp của bỏ ràng buộc lồi |
| A7 | PFAR tĩnh, khớp trên OOF | Đóng góp của G3 |
| A8 | PFAR trực tuyến | Đóng góp của hiệu chỉnh thích ứng |

Dòng **A3 là dòng quan trọng nhất và cũng là dòng nguy hiểm nhất** cho luận văn. Số liệu hiện tại trên SDN: A2 = 5.509, A3 = 4.494, A8 = 4.312. Nghĩa là hiệu chỉnh affine một nhánh đã lấy 18%, còn cả hai nhánh học sâu cộng lại chỉ thêm khoảng 3%.

Bạn phải chủ động trình bày con số này. Có hai cách phản ứng, và chỉ một cách đứng vững được:

- **Không nên:** giấu A3, chỉ so A8 với A5 để khoe mức giảm 34%.
- **Nên:** trình bày đủ, rồi lập luận rằng giá trị của kiến trúc ba nhánh nằm ở **tính bền vững xuyên tập dữ liệu** — trên Abilene, nhánh Global đóng góp quyết định (A8 = 1.620 so với A3 chỉ dựa trên XGBoost sẽ kém hơn nhiều), còn trên SDN thì XGBoost gánh chính. Không có tập nào mà một nhánh duy nhất thắng ở cả ba. Đó là luận điểm trung thực và bảo vệ được.

---

## 10. G7 — Kiểm định thống kê

Thay toàn bộ giao thức cũ:

| Vấn đề cũ | Thay bằng |
|---|---|
| t-test ghép cặp trên 10 run trộn hai pipeline | Chạy lại 10 run trên một pipeline duy nhất |
| t-test với nhánh có phương sai 0.000 | **Wilcoxon signed-rank** |
| 9 kiểm định, hiệu chỉnh Holm chỉ nhắc ở phần thảo luận | Đưa cột `p_holm` vào thẳng bảng |
| Không so sánh chuỗi sai số theo thời gian | Thêm **Diebold–Mariano** trên chuỗi sai số tập Test |

Diebold–Mariano là kiểm định đúng chuẩn cho so sánh độ chính xác dự báo giữa hai mô hình trên cùng chuỗi thời gian, và nó không phụ thuộc vào số lần chạy lại. Với hai mô hình tất định như persistence và XGBoost, đây là kiểm định duy nhất áp dụng được.

Ngoài ra, GWN và BiGRU phải được **chạy lại trên cùng quy trình chia dữ liệu và cùng 10 hạt giống**. Con số trích từ bài báo gốc chỉ được dùng để tham khảo, không được đưa vào bảng so sánh chính như hiện nay.

---

## 11. Bảng kết quả kỳ vọng sau khi hoàn tất

Dựa trên những gì đã đo được (G4 đã kiểm chứng, G2–G3 là kỳ vọng thận trọng):

| Tập | Mô hình cũ | Mô hình mới (đã đo, chưa sửa nhánh) | Sau khi sửa nhánh (kỳ vọng) |
|---|---|---|---|
| SDN | 6.562 ± 0.211 | **4.312 ± 0.042** | 3.9 – 4.3 |
| Géant | 0.796 ± 0.025 | 0.816 ± 0.005 | 0.74 – 0.80 |
| Abilene | 2.731 ± 0.866 | **1.620 ± 0.002** | 1.5 – 1.62 |

Trần lý thuyết cần biết trước để không đuổi theo ảo ảnh: oracle chọn nhánh tốt nhất cho từng mẫu trên SDN là 3.597×10⁻³. Không cơ chế gộp nào — dù thông minh đến đâu — vượt được ngưỡng đó với ba nhánh hiện tại. Khoảng từ 4.31 xuống 3.60 là **toàn bộ** dư địa còn lại của hướng "cải tiến bộ gộp", tức khoảng 16%. Muốn đi xa hơn thì phải cải thiện bản thân các nhánh, không phải cách gộp chúng.

---

## 12. Lịch trình 4 tuần

| Tuần | Giai đoạn | Sản phẩm bàn giao |
|---|---|---|
| 1 | G0 + G1 + chạy lại 10 run sạch | Manifest, bảng baseline, Bảng 3 tin cậy được |
| 1 | G4 chế độ A và B trên cache mới | Con số SDN ≈ 4.3, Abilene ≈ 1.6 |
| 2 | G3 (OOF) + G2 (sửa nhánh Local, đổi loss) | Nhánh Local qua cổng C1–C4 |
| 3 | G5 + G6 | Bảng ablation 8 dòng × 3 tập, phân tích đoạn trôi dạt |
| 4 | G7 + viết lại Mục 2–7 | Bản luận văn v6 |

Thứ tự này là cố ý: **G4 được đưa lên tuần 1** vì nó không cần huấn luyện lại gì cả, chỉ cần thay công thức gộp trên cache có sẵn. Bạn sẽ có con số vượt XGBoost trong vòng vài giờ, trước khi bỏ ra ba tuần cho phần còn lại. Nếu vì lý do nào đó phải dừng sớm, bạn vẫn có kết quả để bảo vệ.

---

## 13. Những gì quy trình này cố tình KHÔNG làm

Liệt kê ra để bạn không bị cám dỗ quay lại:

- **Không tinh chỉnh thêm cổng softmax.** Đã chứng minh trần của nó bằng quy hoạch toàn phương: trên SDN nó chặn ở đúng mức XGBoost. Mọi giờ bỏ vào đó đều lãng phí.
- **Không thêm nhánh thứ tư** trước khi ba nhánh hiện tại đều qua cổng C1–C4. Thêm nhánh vào một hệ đang có nhánh chết chỉ làm bảng ablation khó đọc hơn.
- **Không đổi sang kiến trúc Transformer lớn hơn** cho nhánh Global. Vấn đề trên SDN là số mẫu hiệu dụng, không phải dung lượng mô hình; mô hình lớn hơn sẽ tệ hơn.
- **Không chọn siêu tham số trên tập Test**, kể cả một lần "chỉ để xem". Với khoảng cách ba bậc độ lớn của hệ số ridge giữa các tập, đây là chỗ dễ trượt nhất.
- **Không báo cáo mức giảm phần trăm so với GWN trích từ bài báo** như bằng chứng chính. Chạy lại hoặc hạ xuống mức tham khảo.

---

## 14. Một lưu ý cuối về tính trung thực của luận điểm

Sau khi hoàn tất quy trình này, câu chuyện khoa học của luận văn sẽ đổi, và đổi theo hướng vững hơn:

**Câu chuyện cũ:** "Mô hình đề xuất giảm 17–56% MSE so với GWN trên cả ba tập dữ liệu." — dễ bị đánh đổ, vì persistence cũng đánh bại GWN trên hai trong ba tập.

**Câu chuyện mới:** "Ở tầm dự báo một bước, hai trong ba tập dữ liệu chuẩn (Géant, Abilene) gần như tầm thường — phép sao chép giá trị bước trước đã vượt SOTA đã công bố. Chỉ SDN mới có tín hiệu thực để học. Trên tập đó, cơ chế hiệu chỉnh affine thích ứng theo từng luồng vượt mô hình cây mạnh nhất 21.7% và vượt mô hình gộp lồi 34.3%, đồng thời giảm phương sai giữa các lần chạy 5 lần. Trên Abilene, cùng cơ chế đó giảm 41% sai số và đưa mô hình lần đầu vượt persistence 31%."

Câu chuyện thứ hai vừa đúng, vừa khó phản bác, vừa cho thấy bạn hiểu dữ liệu của mình sâu hơn hầu hết các bài báo trong lĩnh vực này. Đó là thứ hội đồng đánh giá cao hơn nhiều so với một bảng phần trăm đẹp.
