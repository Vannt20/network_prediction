"""
Hệ thống Manifest: khoá dữ liệu và chống lẫn artifact giữa các phiên bản pipeline.

Lỗi B2 (run 0-4 và run 5-9 của Géant/Abilene sinh từ hai pipeline khác nhau, bị cắt
ngầm trong precompute_cache) xảy ra vì không có cơ chế nào đối chiếu artifact với
phiên bản dữ liệu đã sinh ra nó. Module này đóng vai trò đó: mọi artifact ghi ra đĩa
đều kèm manifest, mọi artifact đọc vào đều bị đối chiếu và RAISE nếu lệch.
"""
import os
import json
import hashlib
import subprocess
from dataclasses import dataclass, asdict, fields

PIPELINE_VERSION = "v7"


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
    n_windows_val: int
    n_windows_test: int
    seed: int
    git_commit: str

    # Siêu tham số huấn luyện. Mặc định None vì prepare_feature_store() không biết
    # chúng - chúng chỉ được điền ở run_experiments.py ngay trước khi ghi manifest
    # vào thư mục run (xem with_training_params).
    epochs: int = None
    patience: int = None
    warmup_epochs: int = None
    lr: float = None
    weight_decay: float = None
    # Cấu hình nhánh ML dạng JSON (mô hình, giao thức early stopping, siêu tham số).
    # Nhánh học sâu để None. Seed/số luồng CPU bị loại khỏi chuỗi này vì chúng khác
    # nhau hợp lệ giữa các run.
    model_config: str = None


# Các trường bắt buộc phải trùng khớp tuyệt đối giữa hai artifact.
# 'seed' KHÔNG nằm trong danh sách này: các run khác seed là hợp lệ và cần thiết.
# Khoá mô tả DỮ LIỆU. Luôn được đối chiếu, lệch là raise.
STRICT_KEYS = [
    "pipeline_version", "dataset", "csv_sha256", "n_steps_after_hygiene",
    "split_train", "split_val", "split_test", "seq_len", "k_lags",
    "n_windows_val", "n_windows_test",
]

# Khoá mô tả QUÁ TRÌNH HUẤN LUYỆN. Chỉ đối chiếu khi cả hai phía đều có giá trị,
# vì manifest do prepare_feature_store() sinh ra (và manifest nhúng trong cache)
# không mang thông tin này.
#
# Vì sao cần: logs/localspatialtcn_data_sdn_seq_60/run_4 có 205 epoch trong khi
# trần là 200, tức run đó chạy với --epochs khác các run còn lại. Nếu chỉ đối
# chiếu STRICT_KEYS thì không cách nào phát hiện, và khi chia việc cho hai tài
# khoản Kaggle thì đây đúng là kịch bản lỗi B2 trên một trục khác: các run trông
# như cùng một thí nghiệm nhưng thật ra không phải, và std 10 run mất ý nghĩa.
TRAINING_KEYS = ["epochs", "patience", "warmup_epochs", "lr", "weight_decay", "model_config"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit_short() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def build_manifest(dataset: str, meta: dict, seed: int = 42) -> Manifest:
    """Dựng manifest từ metadata do prepare_feature_store trả về."""
    return Manifest(
        pipeline_version=PIPELINE_VERSION,
        dataset=dataset,
        csv_sha256=meta.get('csv_sha256', 'unknown'),
        n_steps_after_hygiene=int(meta['n_steps_after_hygiene']),
        split_train=int(meta['split_train']),
        split_val=int(meta['split_val']),
        split_test=int(meta['split_test']),
        seq_len=int(meta['seq_len']),
        k_lags=int(meta['k_lags']),
        n_windows_val=int(meta['n_windows_val']),
        n_windows_test=int(meta['n_windows_test']),
        seed=int(seed),
        git_commit=git_commit_short(),
    )


def with_training_params(m, epochs=None, patience=None, warmup_epochs=None,
                         lr=None, weight_decay=None, model_config=None, seed=None):
    """Trả bản sao manifest có thêm siêu tham số huấn luyện.

    Gọi ngay trước save_manifest() trong vòng huấn luyện, nơi các giá trị này mới
    được biết.
    """
    d = (m if isinstance(m, dict) else asdict(m)).copy()
    d.update(epochs=epochs, patience=patience, warmup_epochs=warmup_epochs,
             lr=lr, weight_decay=weight_decay, model_config=model_config)
    if seed is not None:
        d['seed'] = int(seed)
    known = {f.name for f in fields(Manifest)}
    return Manifest(**{k: v for k, v in d.items() if k in known})


def save_manifest(m, dirpath: str) -> None:
    os.makedirs(dirpath, exist_ok=True)
    d = m if isinstance(m, dict) else asdict(m)
    with open(os.path.join(dirpath, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def load_manifest(dirpath: str) -> Manifest:
    path = os.path.join(dirpath, 'manifest.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy manifest tại {path}. Artifact này sinh từ pipeline cũ "
            f"(trước v7) và KHÔNG được dùng. Hãy xoá logs/ và cache/ rồi chạy lại từ đầu."
        )
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    known = {f.name for f in fields(Manifest)}
    return Manifest(**{k: v for k, v in d.items() if k in known})


def assert_compatible(a, b) -> None:
    """Đối chiếu hai manifest trên STRICT_KEYS. Lệch bất kỳ khoá nào -> RAISE.

    Tuyệt đối không được hạ xuống mức cảnh báo rồi đi tiếp: đó chính là cách
    lỗi B2 lọt qua và làm hỏng 5/10 run của Géant và Abilene.
    """
    da = a if isinstance(a, dict) else asdict(a)
    db = b if isinstance(b, dict) else asdict(b)
    for k in STRICT_KEYS:
        if da.get(k) != db.get(k):
            raise RuntimeError(
                f"Artifact không tương thích ở trường '{k}' "
                f"(a={da.get(k)!r}, b={db.get(k)!r}). "
                f"Hãy xoá logs/ và cache/ rồi chạy lại từ đầu."
            )
    # Siêu tham số huấn luyện: chỉ so khi CẢ HAI phía đều có, vì manifest sinh từ
    # prepare_feature_store() và manifest nhúng trong cache không mang thông tin này.
    for k in TRAINING_KEYS:
        va, vb = da.get(k), db.get(k)
        if va is not None and vb is not None and va != vb:
            raise RuntimeError(
                f"Hai run dùng siêu tham số huấn luyện khác nhau ở '{k}' "
                f"(a={va!r}, b={vb!r}). Các run này KHÔNG phải cùng một thí nghiệm "
                f"- không được gộp chung để tính trung bình và độ lệch chuẩn."
            )
