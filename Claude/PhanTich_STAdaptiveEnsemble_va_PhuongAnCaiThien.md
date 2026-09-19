# Phân tích phản biện mã nguồn & kết quả thực nghiệm ST-Adaptive-Ensemble
### Kèm phương án cải thiện theo thứ tự ưu tiên

*Phạm vi kiểm tra: repo `Vannt20/thucnghiem` (toàn bộ mã nguồn + 60 file cache dự đoán + thư mục `logs/`) và báo cáo `NguyenTheVan_BC_ThucNghiem_v5.docx`. Mọi con số dưới đây được tính lại trực tiếp từ file cache trong repo, không lấy từ báo cáo.*

---

## Phần I. Bốn lỗi phải xử lý trước khi bàn đến chuyện cải thiện

Bốn vấn đề dưới đây khiến một phần lớn số liệu trong Bảng 3 và Bảng 4 hiện **không dùng được cho luận văn**. Cần sửa trước, vì nếu không thì mọi cải tiến mô hình sau này đều đo trên một cái thước hỏng.

### 1.1. Nhánh LocalSpatialTCN là baseline persistence trá hình trên Géant và Abilene

Tôi tính khoảng cách giữa đầu ra nhánh Local và phép sao chép giá trị bước liền trước:

| Tập | MSE nhánh Local (×10⁻³) | MSE persistence (×10⁻³) | `mean((y_local[t] − x[t−1])²)` |
|---|---|---|---|
| Géant | 0.854 | 0.855 | **0.000** |
| Abilene | 2.361 | 2.363 | **0.000** |
| SDN | 15.152 | 26.552 | 11.0 |

Trên Géant và Abilene, đầu ra nhánh Local **trùng khít từng chữ số** với giá trị bước thời gian liền trước. Nguyên nhân nằm ở `Graph_models/local_filters.py`:

```python
out_norm = last_val + delta_norm          # Last-Value Skip Connection
nn.init.zeros_(self.out_head[-1].weight)  # delta = 0 tại epoch 0
```

kết hợp với log huấn luyện `logs/localspatialtcn_data_abilene_seq_24/run_5/train_metrics.csv`:

```
epoch,seed,lr,train_loss,val_loss
1,47,0.001,0.012434815922057538,0.01252130417774121   <- best
2,47,...,0.012435054463507112,0.012521377640465896
...
70,47,...,0.012437918836012283,0.012574237162868181   <- early stop
```

`train_loss` **tăng đều đặn từ epoch 1 đến epoch 70**. Checkpoint `best_model.pth` được ghi tại epoch 1, tức mô hình khi `delta ≡ 0`. Mạng không học được gì; early stopping đã giữ lại đúng trạng thái khởi tạo. Điều này giải thích trọn vẹn con số `± 0.000` trong Bảng 3 mà báo cáo đang diễn giải nhầm là "cấu hình gần như tất định".

**Hệ quả với luận văn:** mọi kiểm định t-test có LocalSpatialTCN ở một vế (2 trong 9 dòng Bảng 4) thực chất đang so mô hình đề xuất với persistence, và phải được phát biểu lại như vậy.

### 1.2. Một nửa số run trên Géant và Abilene dùng lại artifact từ phiên bản pipeline cũ

`training/precompute_cache.py` xử lý nhánh Global bất đối xứng: dự đoán trên tập Validation được suy diễn lại từ checkpoint, còn dự đoán trên tập Test thì **nạp lại file `.npy` cũ nếu file tồn tại**, kèm phép cắt ngắn im lặng:

```python
if not quick_check and os.path.exists(y_pred_global_test_file):
    y_glob_test_np = np.load(y_pred_global_test_file)
    if len(y_glob_test_np) == len(x_test_win):
        y_global_test = torch.tensor(y_glob_test_np, ...)
    else:
        y_global_test = torch.tensor(y_glob_test_np[:len(x_test_win)], ...)   # cắt lặng lẽ
```

Kiểm tra thực tế trong `logs/stwaveformer_data_abilene_seq_24/`:

| Run | Số dòng `y_pred_data.npy` | Schema `test_metrics.csv` | MSE trong log | MSE khi đối chiếu với `y_real` trong cache |
|---|---|---|---|---|
| 0–4 | **9596** | không có cột `seed` | 2.45×10⁻³ | **10.38×10⁻³** |
| 5–9 | **9539** | có cột `seed`, `wape`, `raw_*` | 1.99×10⁻³ | 1.98×10⁻³ |

Hai cụm run được sinh ra bởi **hai phiên bản chuẩn bị dữ liệu khác nhau** (độ dài tập test lệch 57 bước, schema log khác nhau). Run 0–4 là tàn dư từ lần chạy trước khi khử 288 mốc trùng lặp của Abilene. Khi `precompute_cache` cắt mảng 9596 dòng xuống 9539 dòng, chuỗi bị lệch trục thời gian và MSE nhánh Global nhảy từ 2.45 lên 10.38.

Con số này khớp chính xác với báo cáo và đó mới là điều đáng lo:

| Đại lượng báo cáo | Giá trị trong báo cáo | Giá trị tôi tính lại từ cache |
|---|---|---|
| Abilene, ST-WaveFormer | 2.219 ± 0.241 | trung bình của (2.45 ×5 run, 1.99 ×5 run) = 2.216 ± 0.240 |
| Géant, ST-WaveFormer | 0.801 ± 0.063 | runs 0–4 = 0.864; runs 5–9 = 0.738 → 0.801 ± 0.063 |

**Độ lệch chuẩn 10 lần chạy trong Bảng 3 không phải phương sai do hạt giống ngẫu nhiên, mà là phương sai do trộn hai phiên bản thí nghiệm.** Kéo theo:

- Khẳng định "Meta-Gating giảm ~60% phương sai trên Géant" mất cơ sở — cái gọi là phương sai đó là artifact kỹ thuật.
- Khẳng định "độ lệch chuẩn Ensemble trên Abilene rất lớn (±0.866) do cổng nhạy với khởi tạo" cũng sai nguyên nhân: 5/10 run được huấn luyện cổng trên một nhánh Global bị hỏng (MSE 10.38 thay vì 1.98).
- Toàn bộ 9 kiểm định t-test ghép cặp ở Bảng 4 đang chạy trên dãy hiệu số không đồng nhất về điều kiện thí nghiệm → cần chạy lại từ đầu.

### 1.3. Hàm mất mát của cổng trong code không phải hàm trong báo cáo

Công thức (13) của báo cáo ghi hàm mất mát của cổng là MSE. `training/train_gate_stacking.py` dòng 98 thực tế dùng:

```python
loss_huber = F.huber_loss(y_hat, y_real_val, delta=0.01)
```

Với `delta = 0.01` trên thang dữ liệu đã chuẩn hoá về [0,1], trong khi MAE điển hình của các nhánh là 0.017–0.067, gần như **mọi mẫu đều rơi vào vùng tuyến tính của Huber**. Nói cách khác, cổng đang được tối ưu theo MAE, còn chỉ số công bố chính lại là MSE. Đây là lệch mục tiêu chứ không chỉ là lỗi trình bày — và nó là một phần lý do cổng không hội tụ về nghiệm tốt cho MSE.

Ba vấn đề phụ cùng nằm trong hàm này:

- Cổng được huấn luyện **full-batch 100 bước** (không phải 100 epoch mini-batch), `lr = 1e-3`, Adam + CosineAnnealing. 100 bước gradient là quá ít để dịch chuyển khỏi điểm khởi tạo.
- Điểm khởi tạo là `bias = −ln(val_mse_k)`. Trên SDN, prior này cho ra softmax ≈ [0.22, 0.21, 0.57]; trọng số cuối cùng báo cáo công bố là 69.7% cho ML. **Cổng gần như chỉ ngồi yên tại prior.**
- `best_state` được chọn theo chính hàm mất mát đang tối ưu trên chính tập Validation, không có tập giữ riêng cho cổng.

### 1.4. XGBoost dùng tập Validation để early-stopping, rồi cổng lại học trọng số trên chính tập đó

`baselines_ml/xgboost_baseline.py` đặt `early_stopping_rounds=30` với `eval_set = [(X_val, y_val)]`. Sau đó `train_gate_stacking.py` huấn luyện cổng trên đúng tập Validation ấy. Nhánh ML vì thế "đẹp" một cách giả tạo trên tập mà cổng dùng để ra quyết định.

Hậu quả đo được rất rõ trên Abilene:

| Nhánh | MSE trên Validation (×10⁻³) | MSE trên Test (×10⁻³) |
|---|---|---|
| Global | 1.121 | 1.978 *(runs 5–9)* |
| Local | 1.436 | 2.361 |
| **ML (XGBoost)** | **0.961 — tốt nhất** | **2.763 — tệ nhất** |

Trọng số lồi tối ưu khớp trên Validation cho ra `w = [0.205, 0.002, 0.793]`: cổng dồn **79% trọng số cho nhánh tệ nhất trên tập Test**. Đây mới là nguyên nhân thật của kết quả kém trên Abilene, không phải "cổng nhạy với khởi tạo ngẫu nhiên".

Vấn đề còn nghiêm trọng hơn ở Géant, nơi phân phối Validation lệch hẳn khỏi Test:

| Tập | MSE Validation (×10⁻³) | MSE Test (×10⁻³) | Tỷ lệ |
|---|---|---|---|
| Géant – Global | 3.292 | 0.801 | 4.1× |
| Géant – Local | 4.949 | 0.854 | 5.8× |
| Géant – ML | 3.144 | 0.915 | 3.4× |

Lát cắt 10% Validation của Géant khó hơn tập Test từ 3 đến 6 lần. Huấn luyện cổng trên lát cắt đó là học một chế độ vận hành không hề xuất hiện trong lúc đánh giá.

---

## Phần II. Vì sao mô hình đề xuất **không thể** thắng XGBoost trên SDN

Đây là câu hỏi trọng tâm. Câu trả lời mang tính toán học chứ không phải vấn đề tinh chỉnh tham số.

Cơ chế gộp hiện tại là **tổ hợp lồi** (softmax ⇒ trọng số không âm, tổng bằng 1). Tôi giải bài toán quy hoạch toàn phương tìm bộ trọng số lồi **tốt nhất có thể** trên chính tập Test (tức oracle, không thể đạt hơn trong thực tế):

**SDN, MSE ×10⁻³ trên tập Test (trung bình 10 run):**

| Cấu hình | MSE | Ghi chú |
|---|---|---|
| Persistence (sao chép bước trước) | 26.552 | baseline còn thiếu trong báo cáo |
| ST-WaveFormer (Global) | 15.350 | |
| LocalSpatialTCN (Local) | 15.152 | |
| Trung bình cộng 1/3 ba nhánh | 9.857 | |
| **ST-Adaptive-Ensemble (báo cáo)** | **6.562** | |
| XGBoost đơn lẻ | 5.509 | |
| Tổ hợp lồi tối ưu khớp trên Val | 5.509 | `w = [0.000, 0.000, 1.000]` |
| **Tổ hợp lồi ORACLE trên Test** | **5.509** | `w = [0.000, 0.000, 1.000]` |

Kết luận không tránh được: **trên SDN, tổ hợp lồi tốt nhất trong vũ trụ chính là "bỏ hẳn hai nhánh học sâu, dùng 100% XGBoost"**. Hai nhánh học sâu kém gấp gần 3 lần và sai số của chúng không đủ phản tương quan với XGBoost để bù lại. Mọi nỗ lực tinh chỉnh cổng softmax đều bị chặn trên bởi con số 5.509 — tức là **hoà, không bao giờ thắng**. Mô hình hiện tại đạt 6.562 vì cổng chưa hội tụ nổi về prior tối ưu.

Nhưng tin tốt: ràng buộc lồi mới là thứ chặn đường, chứ không phải bản thân ý tưởng gộp.

| Cấu hình gộp (SDN) | MSE ×10⁻³ | So với XGBoost |
|---|---|---|
| Tổ hợp lồi ORACLE | 5.509 | 0% |
| Stacking tuyến tính toàn cục (fit Val) | 5.407 | −1.9% |
| **Stacking affine theo từng luồng OD (fit Val)** | **4.403 ± 0.132** | **−20.1%** |
| Oracle chọn nhánh tốt nhất cho từng mẫu | 3.597 | −34.7% *(giới hạn dưới tuyệt đối của mọi cơ chế gating cứng)* |

**Chỉ cần thay công thức gộp — không huấn luyện lại bất kỳ nhánh nào — MSE trên SDN giảm từ 6.562 xuống 4.403, tức vượt XGBoost 20.1%.** Tôi đã chạy phép này trên đúng 10 bộ cache của bạn, khớp Validation và đánh giá trên Test, không có rò rỉ.

Tuy nhiên cần trung thực về nguồn gốc của mức cải thiện đó. Tôi phân rã đóng góp từng nhánh trong stacking theo luồng:

| Tổ hợp nhánh đưa vào stacking | MSE ×10⁻³ (SDN) |
|---|---|
| Chỉ ML + hiệu chỉnh affine theo luồng | **4.494** |
| Global + Local (không có ML) | 14.589 |
| Global + ML | 4.395 |
| Local + ML | 4.351 |
| Cả ba nhánh | 4.403 |

Nghĩa là: **4.494/5.509 ≈ 18% mức cải thiện đến hoàn toàn từ việc hiệu chỉnh độ lệch và hệ số tỷ lệ riêng cho từng luồng OD của XGBoost. Hai nhánh học sâu chỉ đóng góp thêm khoảng 3%.** Nếu bảo vệ luận văn với con số 4.40, hội đồng hoàn toàn có thể hỏi "vậy ba nhánh để làm gì" và câu hỏi đó là chính đáng. Điều này dẫn thẳng tới Phần III.

### Một baseline còn thiếu làm thay đổi cách đọc toàn bộ Bảng 2

| Tập | Persistence (×10⁻³) | MA(3) | GWN công bố | Mô hình đề xuất |
|---|---|---|---|---|
| SDN | 26.552 | 21.588 | 7.936 | 6.562 |
| Géant | **0.855** | 1.013 | 0.879 | 0.796 |
| Abilene | **2.363** | 2.272 | 6.220 | 2.731 |

Trên Géant, phép sao chép giá trị bước liền trước đã tốt hơn GWN đã công bố (0.855 so với 0.879) và chỉ kém mô hình đề xuất 7%. Trên Abilene, persistence tốt hơn GWN 62%, còn mô hình đề xuất lại **kém hơn persistence**. Nói cách khác, mức "giảm 56.10% MSE so với GWN trên Abilene" mà báo cáo nêu ở phần Tóm tắt và Kết luận chủ yếu phản ánh việc bài toán dự báo một bước trên Abilene gần như tầm thường, chứ không phản ánh năng lực của kiến trúc đề xuất.

Đây là điểm dễ bị hội đồng hỏi nhất. Cần chủ động đưa persistence và HA vào Bảng 2 và Bảng 3 rồi thảo luận thẳng, thay vì để người phản biện phát hiện.

Đồng thời điều này cũng đảo ngược cách nhìn của bạn về ba tập dữ liệu: **SDN mới là tập dữ liệu duy nhất mà mô hình học thực sự tạo ra giá trị** (5.5 so với 26.6 của persistence, tức gấp gần 5 lần). Géant và Abilene là hai tập mà mọi mô hình đều dao động quanh persistence. Bạn đang coi SDN là tập "chưa tốt", nhưng xét về mặt khoa học thì SDN mới là nơi luận văn có câu chuyện để kể.

---

## Phần III. Phương án cải thiện, sắp theo tỷ lệ lợi ích trên công sức

### P0 — Làm sạch quy trình thực nghiệm (bắt buộc, khoảng 1 ngày)

Không có bước này thì mọi số đo sau đều không đáng tin.

1. **Xoá toàn bộ `cache/` và `logs/`, chạy lại từ đầu bằng một phiên bản dữ liệu duy nhất.** Đây là việc bắt buộc, vì hiện có hai phiên bản trộn lẫn.
2. **Bỏ cơ chế nạp lại `y_pred_data.npy`** trong `precompute_cache.py`, hoặc chí ít thay phép cắt ngầm bằng khẳng định cứng:
   ```python
   assert len(y_glob_test_np) == len(x_test_win), \
       f"Artifact lệch: {len(y_glob_test_np)} vs {len(x_test_win)} — hãy xoá logs và chạy lại"
   ```
3. **Ghi manifest cho mỗi run**: hash của file CSV đầu vào, `seq_len`, kích thước từng split, commit hash. Có manifest thì lỗi loại 1.2 không bao giờ lặp lại.
4. **Đồng bộ báo cáo với code**: hoặc sửa code sang MSE, hoặc sửa công thức (13) thành Huber và giải thích vì sao.
5. **Tách tập early-stopping của XGBoost ra khỏi tập huấn luyện cổng.** Ví dụ cắt Validation thành Val-A (early stopping cho các nhánh) và Val-B (huấn luyện cổng), hoặc dùng phương án P2 bên dưới.
6. **Thêm persistence và Historical Average vào mọi bảng kết quả.**
7. **Đổi t-test ghép cặp sang Wilcoxon signed-rank** cho các cặp có phương sai suy biến, và báo cáo p hiệu chỉnh Holm–Bonferroni ngay trong bảng chứ không chỉ nói ở phần thảo luận.

### P1 — Thay cổng lồi bằng stacking affine theo từng luồng (lợi ích lớn nhất, khoảng 2 giờ)

Đây là thay đổi đã được kiểm chứng trên chính dữ liệu của bạn: SDN 6.562 → 4.403.

Bản chất: bỏ ràng buộc `w ≥ 0, Σw = 1` và cho phép mỗi luồng OD có bộ hệ số riêng cộng một hệ số chệch:

$$\hat{y}_{t,f} = \alpha_f + \beta_f^G \hat{y}^G_{t,f} + \beta_f^L \hat{y}^L_{t,f} + \beta_f^M \hat{y}^M_{t,f}$$

Cách hiện thực gọn nhất là **giữ nguyên mạng Meta-Gating hiện có nhưng đổi tầng đầu ra**: thay `Softmax` bằng đầu ra tuyến tính 4 chiều (3 hệ số + 1 chệch), cộng thêm một bảng nhúng luồng `nn.Embedding(N, d)` ghép vào vector ngữ cảnh. Như vậy vẫn giữ được tính "thích ứng theo ngữ cảnh" của ý tưởng gốc — điểm bán hàng của luận văn — mà không bị chặn trên bởi tổ hợp lồi.

Những chi tiết cần đi kèm:

- Chính quy hoá ridge, `λ ∈ [1e-5, 1e-3]`; tôi đã quét và `1e-4` là điểm an toàn trên cả ba tập.
- Ràng buộc mềm về tính lồi (ví dụ phạt `‖Σβ − 1‖²`) nếu muốn giữ khả năng diễn giải; mất khoảng 0.05e-3 nhưng bảng trọng số vẫn đọc được.
- Tăng số bước huấn luyện cổng: 100 bước full-batch là quá ít. Dùng mini-batch, 2000–5000 bước, early stopping trên một tập giữ riêng.
- Đổi `huber_loss(delta=0.01)` thành `mse_loss` nếu chỉ số công bố chính là MSE.

**Lưu ý quan trọng:** trên Abilene, stacking theo luồng khớp trên Validation cho 2.314e-3, vẫn **kém hơn** nhánh Global đơn lẻ (1.978e-3). Nguyên nhân là lệch phân phối Val↔Test đã nêu ở mục 1.4. Vì vậy P1 phải đi kèm P2 mới hoàn chỉnh.

### P2 — Huấn luyện bộ gộp trên dự đoán out-of-fold của tập Train, không phải trên lát cắt Validation (khoảng 1 ngày)

Đây là sửa chữa mang tính nguyên lý, và là thứ nên viết thành đóng góp khoa học của luận văn.

Vấn đề gốc: bộ gộp đang học từ 10% dữ liệu nằm trong một khoảng thời gian duy nhất, nên nó học đặc tính của **một chế độ vận hành mạng**, rồi đem áp lên tập Test có chế độ khác. Bảng lệch Val/Test ở mục 1.4 cho thấy tỷ lệ lệch lên tới 4–6 lần.

Cách làm chuẩn:

1. Chia tập Train thành K fold liên tiếp theo thời gian (blocked time-series CV, K = 5).
2. Với mỗi fold, huấn luyện ba nhánh trên các fold trước, sinh dự đoán cho fold đang xét → thu được ma trận dự đoán **out-of-fold phủ toàn bộ tập Train**.
3. Huấn luyện bộ gộp trên toàn bộ ma trận OOF này (gấp 7 lần dữ liệu, phủ nhiều chế độ vận hành).
4. Giữ Validation thuần tuý cho early stopping, Test hoàn toàn không đụng tới.

Chi phí là phải huấn luyện ba nhánh K lần thay vì 1 lần. Với Abilene (48k bước) thì đáng; với SDN (6.2k bước) thì gần như miễn phí.

### P3 — Bổ sung hiệu chỉnh trực tuyến cho bộ gộp (khoảng nửa ngày, và đây là luận điểm khoa học mạnh nhất còn lại)

Báo cáo ở Mục 6 đã tự đặt đúng giả thuyết: lợi thế của cơ chế gộp nằm ở khả năng thích ứng khi phân phối thay đổi. Nhưng hiện chưa có bằng chứng nào, mà kịch bản đánh giá tĩnh lại đang triệt tiêu chính lợi thế đó.

Đề xuất cụ thể: thêm vào vector ngữ cảnh bốn chiều hiện tại (`σ_local`, `spike_flag`, `tod`, `dow`) ba nhóm tín hiệu mà cổng đang hoàn toàn mù:

- **Sai số trượt gần đây của từng nhánh trên từng luồng**: `EWMA(e²_{t−1,f})` cho mỗi nhánh. Đây là tín hiệu quan trọng nhất và hiện đang thiếu hẳn — cổng không hề biết nhánh nào vừa dự báo sai.
- **Độ bất đồng giữa các nhánh**: `|ŷ^G − ŷ^M|`, `|ŷ^L − ŷ^M|`, phương sai ba dự báo. Bất đồng cao là chỉ báo kinh điển cho việc nên thận trọng.
- **Nhúng danh tính luồng** (`nn.Embedding`), để cổng học được rằng luồng nào hợp với nhánh nào.

Kèm theo đó là kịch bản đánh giá trực tuyến: cập nhật bộ gộp theo cửa sổ trượt trên tập Test (chỉ dùng thông tin quá khứ, không rò rỉ), báo cáo song song với kết quả tĩnh. Nếu kết quả trực tuyến vượt rõ rệt XGBoost tĩnh, bạn có một đóng góp khoa học sạch sẽ và bảo vệ được — thay vì phải biện minh cho một mô hình phức tạp mà thua một mô hình cây đơn giản.

Lưu ý giới hạn: oracle chọn nhánh tốt nhất cho từng mẫu trên SDN là 3.597e-3. Không cơ chế cổng nào vượt được ngưỡng đó. Khoảng cách từ 4.40 xuống 3.60 là toàn bộ dư địa còn lại của hướng "cổng thông minh hơn", tức khoảng 18%. Đừng kỳ vọng nhiều hơn.

### P4 — Hồi sinh hai nhánh học sâu (1–2 ngày, rủi ro trung bình)

Cần làm cho đủ tính chính danh của kiến trúc ba nhánh, dù lợi ích số học có thể khiêm tốn.

**Nhánh Local (ưu tiên cao, lỗi rõ ràng):**
- Bỏ `nn.init.zeros_` ở tầng cuối, hoặc giữ zero-init nhưng **gỡ early stopping trong 10 epoch đầu**, để mô hình thoát khỏi điểm khởi tạo persistence trước khi bị chấm điểm.
- Ghi log `mean(|delta_norm|)` mỗi epoch. Nếu đại lượng này ở mức 1e-8 thì nhánh đã chết và cần dừng chạy ngay, thay vì để nó chạy hết 70 epoch rồi mới phát hiện.
- Tăng learning rate cho riêng `out_head` (ví dụ 10×), vì gradient đi qua tầng zero-init rất nhỏ ở giai đoạn đầu.
- Xem lại `weight_decay=1e-4` áp lên cả `out_head`: với một tầng khởi tạo bằng 0, weight decay kéo nó về lại 0.

**Nhánh Global:**
- Đổi `SmoothL1Loss(beta=0.01)` sang MSE, hoặc ít nhất tăng `beta` lên cỡ 0.05–0.1. Hiện gần như mọi mẫu rơi vào vùng tuyến tính, tức mạng đang tối ưu MAE trong khi bạn công bố MSE. Trên SDN nhiều gai xung, khác biệt giữa nghiệm trung vị và nghiệm trung bình là rất lớn.
- Trên Géant/Abilene, nhánh Global cũng chỉ lệch khỏi persistence rất ít (`mean((ŷ^G[t] − x[t−1])²) = 0.175e-3` trên Géant runs 5–9, so với MSE 0.738e-3). Nó cũng đang gần như sao chép. Cần thí nghiệm dự báo **phần dư so với persistence** thay vì dự báo mức tuyệt đối, và đánh giá bằng chỉ số tương đối với persistence (skill score) để thấy rõ mô hình có học được gì thật không.

**Về khoảng cách trên SDN (15.2 so với 5.5):** giả thuyết đáng kiểm tra nhất là bất đối xứng về số mẫu hiệu dụng. Tập SDN chỉ có 4,380 bước huấn luyện, tức 4,320 cửa sổ trượt cho các nhánh học sâu; trong khi XGBoost ở dạng shared model nhìn thấy 4,320 × 196 ≈ 846 nghìn dòng. Hướng xử lý là chuyển các nhánh học sâu sang chế độ **channel-independent** (mỗi luồng là một mẫu, chia sẻ trọng số — cách làm của DLinear/PatchTST), giữ tương tác không gian ở một tầng nhẹ phía sau. Đây cũng là một thí nghiệm rẻ và có thể trở thành một mục ablation tốt.

### P5 — Đổi mô hình gộp từ song song sang nối tiếp (nếu P1–P4 chưa đủ)

Nếu sau các bước trên hai nhánh học sâu vẫn đóng góp dưới 5%, nên thẳng thắn đổi kiến trúc thay vì cố bảo vệ sơ đồ ba nhánh song song:

- **Cascade dư**: XGBoost dự báo mức, các nhánh học sâu học phần dư `y − ŷ^M` theo cấu trúc không gian–thời gian. Cách này khai thác đúng thế mạnh của từng họ mô hình và thường vượt gộp song song rõ rệt khi một nhánh mạnh hơn hẳn.
- **Hoặc**: dùng dự đoán của hai nhánh học sâu làm **đặc trưng đầu vào** cho XGBoost (feature-level fusion thay vì prediction-level fusion). Rẻ, dễ cài, và thường hiệu quả hơn cả stacking.

---

## Phần IV. Danh mục chỉnh sửa cần thực hiện trong báo cáo

| Mục | Nội dung hiện tại | Cần sửa thành |
|---|---|---|
| Tóm tắt, Mục 7 | "giảm 56.10% MSE trên Abilene so với GWN" | Nêu kèm persistence = 2.363e-3 (tốt hơn GWN 62%) và ghi rõ mô hình đề xuất **kém hơn** persistence trên tập này |
| Mục 4.2, Mục 7 | "giảm ~60% phương sai trên Géant" | Bỏ hoặc phát biểu lại sau khi chạy lại 10 run trên một pipeline thống nhất |
| Bảng 3, dòng LocalSpatialTCN | "±0.000, cấu hình gần như tất định" | Ghi rõ nhánh này hội tụ về persistence; hoặc huấn luyện lại theo P4 |
| Công thức (13) | `L_gate = MSE` | Huber với δ = 0.01, hoặc đổi code sang MSE |
| Bảng 4 | 9 kiểm định t-test | Chạy lại sau P0; dùng Wilcoxon cho cặp có phương sai suy biến; đưa p hiệu chỉnh Holm vào thẳng bảng |
| Bảng 2, Bảng 3 | Thiếu baseline naive | Thêm dòng Persistence và Historical Average cho cả ba tập |
| Mục 3.1 | Chỉ MSE/MAE/RMSE trên thang chuẩn hoá | Thêm chỉ số skill score so với persistence — đây mới là thước đo cho thấy mô hình học được gì |
| Mục 6, gạch đầu dòng 3 | Thời gian suy diễn 0.00–0.02 ms | Đo lại tổng ba nhánh + tầng gộp trên cùng một điều kiện; báo cáo p50/p95 |

---

## Phần V. Lộ trình đề xuất

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 1 | P0 toàn bộ: xoá cache/logs, chạy lại 10 run trên một pipeline, thêm baseline naive, thêm manifest | Bảng 3 và Bảng 4 mới, tin cậy được |
| 1 | P1: đổi cổng sang stacking affine theo luồng, chạy trên cache mới | Kỳ vọng SDN ≈ 4.4e-3, vượt XGBoost ~20% |
| 2 | P2: sinh dự đoán OOF trên tập Train, huấn luyện lại bộ gộp | Kỳ vọng sửa được điểm yếu Abilene |
| 2 | P4: hồi sinh nhánh Local (bỏ early stopping sớm, đổi loss), kiểm tra `mean|delta|` | Bảng ablation có ý nghĩa |
| 3 | P3: mở rộng vector ngữ cảnh + kịch bản đánh giá trực tuyến | Bằng chứng cho luận điểm "thích ứng khi phân phối thay đổi" |
| 4 | Viết lại Mục 4–7, bổ sung bảng ablation đầy đủ | Bản luận văn hoàn chỉnh |

---

## Ghi chú về cách đọc kết quả này

Mọi con số trong tài liệu này được tính lại từ 60 file cache trong repo bằng một trình đọc `.pt` độc lập, không chạy lại huấn luyện. Bộ trọng số lồi tối ưu được giải bằng quy hoạch toàn phương trên đơn hình (liệt kê đủ 7 mặt), không phải bằng gradient descent, nên là nghiệm chính xác. Stacking theo luồng được khớp **chỉ trên tập Validation** và đánh giá trên tập Test — cùng giao thức với mô hình hiện tại, không có rò rỉ.

Con số Abilene trong Phần II được tính riêng trên các run 5–9 vì các run 0–4 chứa artifact hỏng đã nêu ở mục 1.2.
