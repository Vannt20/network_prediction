"""
Commit + push artifact của các run đã hoàn tất. Gọi sau mỗi run trên Kaggle.

    python tools/push_run.py --account A --note "sdn STWaveFormer run 0"
    python tools/push_run.py --account A --pull-first     # gộp việc của account kia

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
    ap.add_argument('--pull-first', action='store_true',
                    help="Kéo và rebase trước khi push - dùng khi hai account cùng "
                         "đẩy lên một nhánh.")
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

    if sh(['git', 'remote'], quiet=True).stdout.strip():
        if a.pull_first:
            sh(['git', 'pull', '--rebase'], check=False)
        cur = sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], quiet=True).stdout.strip()
        sh(['git', 'push', '-u', 'origin', cur], check=False)
    else:
        print("Chưa cấu hình remote - đã commit cục bộ, chưa push.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
