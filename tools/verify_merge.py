"""
Kiểm tra tính đồng nhất của artifact trước khi gộp kết quả từ nhiều nguồn.

BẮT BUỘC chạy sau khi kéo kết quả từ hai tài khoản Kaggle về cùng một repo.

Vì sao. Lỗi B2 của pipeline cũ có cấu trúc y hệt kịch bản chạy song song: run 0-4
và run 5-9 của Géant/Abilene được sinh ra bởi hai phiên bản chuẩn bị dữ liệu khác
nhau (độ dài tập test lệch 57 bước), rồi bị cắt ngầm khi gộp. Hậu quả: MSE nhánh
Global nhảy từ 1.98e-3 lên 10.38e-3, và toàn bộ độ lệch chuẩn 10 run trong báo cáo
trở thành artifact kỹ thuật chứ không phải phương sai do hạt giống.

Chia việc giữa hai tài khoản tái tạo đúng rủi ro đó. Script này là chốt chặn.

    python tools/verify_merge.py                    # kiểm tra logs/ và cache/
    python tools/verify_merge.py --expect_runs 10   # thêm: đủ 10 run mỗi model
"""
import os
import sys
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        try:
            _s.reconfigure(encoding='utf-8')
        except Exception:
            pass
import glob
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.manifest import STRICT_KEYS, TRAINING_KEYS

DATASETS = ['sdn', 'geant', 'abilene']


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_log_manifests(logs_dir='logs'):
    """Gom manifest của mọi run trong logs/, nhóm theo (dataset, thư mục model)."""
    groups = defaultdict(list)
    for mf_path in glob.glob(os.path.join(logs_dir, '*', 'run_*', 'manifest.json')):
        parts = mf_path.replace('\\', '/').split('/')
        model_dir, run_dir = parts[-3], parts[-2]
        try:
            mf = _load_json(mf_path)
        except Exception as e:
            groups[('<unreadable>', model_dir)].append((run_dir, {'error': str(e)}))
            continue
        groups[(mf.get('dataset', '?'), model_dir)].append((run_dir, mf))
    return groups


def collect_cache_manifests(cache_dir='cache'):
    import torch
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(cache_dir, '*_preds.pt'))):
        name = os.path.basename(f)
        ds = name.split('_run_')[0]
        d = torch.load(f, weights_only=False)
        groups[ds].append((name, d.get('manifest')))
    return groups


def check_group(label, items, expect_runs=None):
    """Đối chiếu mọi manifest trong một nhóm với manifest đầu tiên."""
    problems = []
    if not items:
        return problems

    missing = [n for n, mf in items if not mf or 'error' in (mf or {})]
    for n in missing:
        problems.append(f"{label}/{n}: thiếu hoặc không đọc được manifest "
                        f"(artifact sinh từ pipeline cũ - phải chạy lại)")

    valid = [(n, mf) for n, mf in items if mf and 'error' not in mf]
    if not valid:
        return problems

    ref_name, ref = valid[0]
    for name, mf in valid[1:]:
        for k in STRICT_KEYS:
            if mf.get(k) != ref.get(k):
                problems.append(
                    f"{label}: '{name}' lệch '{ref_name}' ở trường '{k}' "
                    f"({mf.get(k)!r} vs {ref.get(k)!r})")
        # Siêu tham số huấn luyện: chỉ so khi cả hai phía cùng có giá trị.
        for k in TRAINING_KEYS:
            va, vb = mf.get(k), ref.get(k)
            if va is not None and vb is not None and va != vb:
                problems.append(
                    f"{label}: '{name}' lệch '{ref_name}' ở siêu tham số huấn luyện "
                    f"'{k}' ({va!r} vs {vb!r}) - KHÔNG phải cùng một thí nghiệm")

    # Cảnh báo (không phải lỗi) nếu run thiếu hẳn siêu tham số huấn luyện.
    no_tp = [n for n, mf in valid if all(mf.get(k) is None for k in TRAINING_KEYS)]
    if no_tp and len(no_tp) != len(valid):
        problems.append(
            f"{label}: {len(no_tp)} run không ghi siêu tham số huấn luyện "
            f"({', '.join(no_tp[:4])}{'...' if len(no_tp) > 4 else ''}) trong khi "
            f"các run khác có - chúng sinh từ phiên bản code cũ, phải chạy lại")

    # Seed phải ĐÔI MỘT KHÁC NHAU. Hai tài khoản cùng chạy run 0-4 (thay vì chia
    # 0-4 / 5-9) sẽ cho ra các run trùng seed - std tính trên đó là vô nghĩa.
    seeds = [mf.get('seed') for _, mf in valid]
    dup = {s for s in seeds if seeds.count(s) > 1}
    if dup:
        problems.append(f"{label}: seed trùng lặp {sorted(dup)} - hai nguồn chạy "
                        f"cùng một dải run. Phải chia run_ids rời nhau.")

    if expect_runs is not None and len(valid) != expect_runs:
        problems.append(f"{label}: có {len(valid)} run, kỳ vọng {expect_runs}")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs_dir', default='logs')
    ap.add_argument('--cache_dir', default='cache')
    ap.add_argument('--expect_runs', type=int, default=None)
    a = ap.parse_args()

    all_problems = []

    print("=" * 78)
    print("ĐỐI CHIẾU MANIFEST TRONG logs/")
    print("=" * 78)
    groups = collect_log_manifests(a.logs_dir)
    if not groups:
        print("  (không tìm thấy manifest.json nào - logs/ rỗng hoặc từ pipeline cũ)")
    for (ds, model_dir), items in sorted(groups.items()):
        label = f"{model_dir}"
        probs = check_group(label, items, a.expect_runs)
        seeds = sorted(mf.get('seed') for _, mf in items if mf and 'seed' in mf)
        tp = {tuple(mf.get(k) for k in TRAINING_KEYS)
              for _, mf in items if mf and 'error' not in mf}
        status = 'OK' if not probs else 'LỖI'
        print(f"  [{status:4}] {label:<45} {len(items):>2} run, seed={seeds}")
        for t in sorted(tp, key=str):
            if any(v is not None for v in t):
                print(f"         epochs={t[0]} patience={t[1]} warmup={t[2]} "
                      f"lr={t[3]} wd={t[4]}")
        all_problems += probs

    print()
    print("=" * 78)
    print("ĐỐI CHIẾU MANIFEST TRONG cache/")
    print("=" * 78)
    try:
        cgroups = collect_cache_manifests(a.cache_dir)
    except Exception as e:
        cgroups = {}
        print(f"  (bỏ qua: {e})")
    if not cgroups:
        print("  (chưa có cache - chạy training/precompute_cache.py sau khi gộp)")
    for ds, items in sorted(cgroups.items()):
        probs = check_group(f"cache:{ds}", items)
        nwt = {mf.get('n_windows_test') for _, mf in items if mf}
        status = 'OK' if not probs else 'LỖI'
        print(f"  [{status:4}] {ds:<45} {len(items):>2} file, n_windows_test={nwt}")
        all_problems += probs

    print()
    print("=" * 78)
    if all_problems:
        print(f"PHÁT HIỆN {len(all_problems)} VẤN ĐỀ - KHÔNG ĐƯỢC GỘP:")
        for p in all_problems:
            print(f"  - {p}")
        print()
        print("Cách xử lý: xoá các run có manifest lệch rồi chạy lại chúng trên đúng")
        print("phiên bản dữ liệu và code. Tuyệt đối không cắt/đệm mảng để 'khớp kích")
        print("thước' - đó chính là nguyên nhân lỗi B2.")
        return 1

    print("TẤT CẢ ARTIFACT TƯƠNG THÍCH - an toàn để gộp và chạy precompute_cache.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
