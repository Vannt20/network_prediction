"""
Commit + push artifact của các run đã hoàn tất. Gọi sau mỗi run trên Kaggle.

    python tools/push_run.py --account A --note "sdn STWaveFormer run 0"

Trước khi push luôn fetch và merge phần của nhánh trên remote (nếu có) - để chạy
lại notebook nhiều lần trên cùng một tài khoản không làm lịch sử bị tách nhánh.
Push thất bại thì trả mã khác 0 để notebook dừng lại, không đi tiếp trong im lặng.

Chỉ đưa lên những file nhẹ và cần thiết (xem .gitignore):
    best_model.pth (~0.6 MB/run) + train_metrics.csv + test_metrics.csv
    + manifest.json + gates.json

Các file .npy (~22 MB/run) và cache/ (~1.3 GB) bị loại vì tái tạo được, và vì v7
đã bỏ cơ chế nạp lại y_pred_data.npy - nguồn gốc lỗi B2.

Vì sao mỗi run một commit: session Kaggle bị ngắt ở giờ thứ 11 là chuyện thường.
Commit theo từng run nghĩa là mất nhiều nhất một run, thay vì mất cả session.
"""
import os
import sys
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        try:
            _s.reconfigure(encoding='utf-8')
        except Exception:
            pass
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sh(args, check=True, quiet=False):
    r = subprocess.run(args, capture_output=True, text=True)
    if not quiet and r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()
        if check:
            raise RuntimeError(f"$ {' '.join(args)}\n{msg}")
        if not quiet:
            print(f"  [!] {msg}")
    return r


def count_runs():
    """Đếm số run có manifest, để đưa vào thông điệp commit."""
    import glob
    return len(glob.glob(os.path.join('logs', '*', 'run_*', 'manifest.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--account', default='', help="Nhãn nguồn, ví dụ A hoặc B")
    ap.add_argument('--note', default='', help="Mô tả ngắn phần việc vừa xong")
    ap.add_argument('--branch', default=None,
                    help="Nhánh đích. Mặc định: runs/<account> nếu có --account, "
                         "nếu không thì nhánh hiện tại.")
    ap.add_argument('--verify', action='store_true',
                    help="Chạy tools/verify_merge.py trước khi commit; dừng nếu lỗi.")
    a = ap.parse_args()

    if not os.path.isdir('.git'):
        print("Chưa có repo git. Chạy trước:\n"
              "  git init && git add -A && git commit -m 'v7 pipeline'\n"
              "  git remote add origin <URL> && git push -u origin main")
        return 1

    if a.verify:
        r = subprocess.run([sys.executable, 'tools/verify_merge.py'])
        if r.returncode != 0:
            print("\n[DỪNG] verify_merge phát hiện artifact không tương thích. "
                  "Không commit.")
            return 1

    branch = a.branch or (f"runs/{a.account}" if a.account else None)
    if branch:
        cur = sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], quiet=True).stdout.strip()
        if cur != branch:
            # Tạo nhánh nếu chưa có, ngược lại chuyển sang
            if sh(['git', 'rev-parse', '--verify', branch],
                  check=False, quiet=True).returncode != 0:
                sh(['git', 'checkout', '-b', branch])
            else:
                sh(['git', 'checkout', branch])

    sh(['git', 'add', '-A'])

    st = sh(['git', 'status', '--porcelain'], quiet=True).stdout.strip()
    if not st:
        print("Không có thay đổi nào để commit.")
        return 0
    print(f"{len(st.splitlines())} file thay đổi.")

    who = f"[{a.account}] " if a.account else ""
    note = a.note or "artifact training"
    msg = f"{who}{note} ({count_runs()} run có manifest, {time.strftime('%Y-%m-%d %H:%M')})"
    sh(['git', 'commit', '-m', msg])
    print(f"Đã commit: {msg}")

    if not sh(['git', 'remote'], quiet=True).stdout.strip():
        print("Chưa cấu hình remote - đã commit cục bộ, chưa push.")
        return 0

    cur = sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], quiet=True).stdout.strip()
    return push_with_sync(cur)


def push_with_sync(branch):
    """Đồng bộ với nhánh trên remote rồi push. Trả mã khác 0 nếu thất bại.

    Tình huống có thật: tài khoản A chạy một session, push lên runs/A, rồi chạy
    lại notebook. Session mới clone `main` và tạo runs/A MỚI từ đó, nên lịch sử
    tách khỏi runs/A trên remote và push bị từ chối ("fetch first").

    Trước đây hàm này push với check=False rồi trả 0, nên notebook đi tiếp như
    thể mọi thứ ổn. Giờ: thử merge phần của remote vào (merge, không rebase - các
    file .pth là nhị phân nên rebase qua từng commit dễ vỡ hơn), và nếu vẫn không
    được thì DỪNG với hướng dẫn cụ thể.
    """
    # Fetch TƯỜNG MINH đúng nhánh này. `git fetch origin` là không đủ: bản clone
    # nông (`git clone --depth N`) ngầm bật --single-branch, refspec chỉ còn main,
    # nên origin/<branch> không bao giờ được tạo và việc đồng bộ bị bỏ qua âm thầm.
    # Nhánh chưa tồn tại trên remote thì lệnh này lỗi - vô hại, bỏ qua.
    sh(['git', 'fetch', 'origin', f'+refs/heads/{branch}:refs/remotes/origin/{branch}'],
       check=False, quiet=True)
    remote_ref = f'origin/{branch}'
    has_remote = sh(['git', 'rev-parse', '--verify', remote_ref],
                    check=False, quiet=True).returncode == 0

    if has_remote:
        behind = sh(['git', 'merge-base', '--is-ancestor', remote_ref, 'HEAD'],
                    check=False, quiet=True).returncode != 0
        if behind:
            print(f"Nhánh {remote_ref} có commit mà bản này chưa có - thử merge vào...")
            r = sh(['git', 'merge', '--no-edit', remote_ref], check=False)
            if r.returncode != 0:
                sh(['git', 'merge', '--abort'], check=False, quiet=True)
                print(f"\n[DỪNG] Không merge được {remote_ref}: hai phía cùng sửa một file")
                print("(thường là cùng một thư mục run_* được huấn luyện ở hai session).")
                print("Commit đã nằm an toàn ở bản cục bộ. Chọn một trong hai:\n")
                print("  1) Giữ bản MỚI, sao lưu bản trên remote sang nhánh khác:")
                print(f"     git push origin {remote_ref}:refs/heads/{branch}-backup")
                print(f"     git push --force-with-lease origin {branch}\n")
                print("  2) Giữ bản trên remote, bỏ commit này:")
                print(f"     git reset --hard {remote_ref}")
                return 1

    r = sh(['git', 'push', '-u', 'origin', branch], check=False)
    if r.returncode != 0:
        print(f"\n[DỪNG] Push {branch} thất bại. Commit vẫn nằm ở bản cục bộ; xem lỗi ở trên.")
        return 1
    print(f"Đã push {branch}.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
