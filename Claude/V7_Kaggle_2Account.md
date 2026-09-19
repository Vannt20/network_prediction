# Chạy v7 trên 2 tài khoản Kaggle song song + đồng bộ qua git

## 1. Nguyên tắc chia việc

**Chia theo `run_id`, không chia theo dataset.**

Chia theo dataset để Abilene một mình một account là sai: Abilene chiếm quá nửa
tổng chi phí (33,441 cửa sổ, gấp 8 lần SDN), nên nó thành nút thắt và account kia
ngồi chơi. Chia theo run thì cả hai cùng về đích.

| | Account A | Account B |
|---|---|---|
| run_ids | `0,1,2,3,4` | `5,6,7,8,9` |
| seed tương ứng | 42, 43, 44, 45, 46 | 47, 48, 49, 50, 51 |
| nhánh git | `runs/A` | `runs/B` |

`SEEDS` được đánh chỉ số theo `run_id` (`run_experiments.py:102`), nên run 7 luôn
dùng seed 49 bất kể nó chạy ở máy nào. Đây là điều kiện cần để gộp kết quả.

> **Rủi ro phải biết trước.** Kịch bản "run 0–4 ở một nguồn, run 5–9 ở nguồn khác"
> chính là cấu trúc của lỗi B2 trong pipeline cũ: hai nửa được sinh bởi hai phiên
> bản dữ liệu khác nhau, rồi bị cắt ngầm khi gộp, đẩy MSE nhánh Global từ 1.98e-3
> lên 10.38e-3 và biến toàn bộ std 10 run thành artifact kỹ thuật.
>
> Chốt chặn: cả hai account **clone cùng một commit** (nên `csv_sha256` giống
> nhau), mỗi run ghi `manifest.json`, và `tools/verify_merge.py` phải PASS trước
> khi chạy `precompute_cache`. Không được bỏ qua bước này.

## 2. Thiết lập git (làm một lần, trên máy local)

```bash
git init
git add -A
git commit -m "v7 pipeline"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

`.gitignore` đã cấu hình để repo chỉ còn **~186 MB** thay vì 3.6 GB:

| | |
|---|---|
| ĐƯA LÊN | `data/*.csv` (139 MB) — phải giống hệt nhau giữa hai account, vì `csv_sha256` nằm trong manifest |
| | `logs/**/best_model.pth` (43 MB tổng) — đầu vào bắt buộc của `precompute_cache` |
| | `logs/**/*.csv`, `manifest.json`, `gates.json` (0.4 MB) |
| BỎ | `logs/**/*.npy` (2,155 MB) — v7 đã bỏ cơ chế reuse nên chúng không còn là đầu vào của gì cả |
| | `cache/` (1,308 MB) — tái tạo trong vài phút trên CPU từ checkpoint |

`data/abilene.csv` là 67 MB, trên ngưỡng cảnh báo 50 MB của GitHub nhưng dưới giới
hạn cứng 100 MB — push được, chỉ bị cảnh báo. Nếu muốn repo gọn hơn thì đưa `data/`
lên Kaggle Dataset và share cho cả hai account; đổi lại phải tự đảm bảo hai bên
dùng đúng một bản, thứ mà git làm sẵn cho bạn.

## 3. Notebook Kaggle (giống nhau cho cả hai account, chỉ khác `ACCOUNT`)

```python
# ==== CELL 1: cấu hình ====
ACCOUNT  = "A"                      # Account B đổi thành "B"
RUN_IDS  = "0,1,2,3,4"              # Account B: "5,6,7,8,9"
DATASET  = "sdn"                    # chạy lần lượt: sdn -> geant -> abilene
REPO     = "https://<TOKEN>@github.com/<user>/<repo>.git"   # token trong Kaggle Secrets

# ==== CELL 2: lấy code ====
!git clone --depth 50 $REPO /kaggle/working/repo
%cd /kaggle/working/repo
!git checkout -B runs/$ACCOUNT
!git config user.email "you@example.com" && git config user.name "kaggle-$ACCOUNT"
!pip install -q -r requirements.txt

# ==== CELL 3: HIỆU CHỈNH TRƯỚC — bắt buộc, ~10 phút ====
# Ước lượng thời gian hiện đang bất định 25-50 lần giữa hai cách tính.
# Chạy 3 epoch, đo, rồi mới quyết định phạm vi.
!python run_experiments.py --dataset abilene --model STWaveFormer --runs 1 --epochs 3

# ==== CELL 4: training ====
!python run_experiments.py --dataset $DATASET --model STWaveFormer \
    --run_ids $RUN_IDS --skip_existing
!python tools/push_run.py --account $ACCOUNT --note "$DATASET STWaveFormer $RUN_IDS"

!python run_experiments.py --dataset $DATASET --model LocalSpatialTCN \
    --run_ids $RUN_IDS --skip_existing
!python tools/push_run.py --account $ACCOUNT --note "$DATASET LocalSpatialTCN $RUN_IDS"

!python -m baselines_ml.run_ml_baselines --dataset $DATASET --runs 10
!python tools/push_run.py --account $ACCOUNT --note "$DATASET ML baselines"
```

**`--skip_existing` là bắt buộc.** Session Kaggle bị ngắt ở giờ thứ 11 là chuyện
thường; chạy lại notebook sẽ bỏ qua các run đã xong. Trong v7 nó **đối chiếu
manifest trước khi bỏ qua** — run cũ có manifest lệch thì chạy lại chứ không skip.
Bỏ qua mù chính là cách lỗi B2 lọt vào bộ kết quả cuối.

**Push sau mỗi model, không phải cuối session.** Mất nhiều nhất một model thay vì
cả session.

## 4. Gộp kết quả (trên máy local, CPU)

```bash
git fetch origin
git checkout main
git merge origin/runs/A origin/runs/B      # không đụng nhau: khác thư mục run_*
```

```bash
python tools/verify_merge.py --expect_runs 10
```

Script này đối chiếu mọi `manifest.json`, và bắt ba loại lỗi:
- artifact thiếu manifest (sinh từ pipeline cũ),
- lệch bất kỳ trường nào trong `STRICT_KEYS` (đúng lỗi B2),
- **seed trùng lặp** — dấu hiệu hai account cùng chạy một dải run, khiến `std` 10
  run trở nên vô nghĩa.

Chỉ khi nó in `TẤT CẢ ARTIFACT TƯƠNG THÍCH` mới được chạy tiếp:

```bash
python training/precompute_cache.py --datasets all --runs 10
python run_pipeline_v7.py --stage combine --dataset all --runs 10
```

## 5. Thứ tự dataset

SDN → Géant → Abilene. SDN nhẹ nhất nên nếu có gì sai trong cấu hình thì phát hiện
sớm; Abilene nặng nhất nên để sau cùng, và nếu cháy quota thì vẫn còn hai tập có
kết quả đầy đủ.
