"""
Chạy nhiều run song song, mỗi GPU một tiến trình. Dành cho Kaggle "GPU T4 x2".

    python tools/run_multi_gpu.py --dataset sdn --model STWaveFormer --run_ids 0,1,2,3,4

Kết hợp với việc chia run giữa hai tài khoản Kaggle: đây là mức chia THỨ HAI, lồng
bên trong mức thứ nhất. Mỗi tài khoản chỉ truyền phần run của mình:

    Account A  --run_ids 0,1,2,3,4  ->  GPU0:[0,2,4]  GPU1:[1,3]
    Account B  --run_ids 5,6,7,8,9  ->  GPU0:[5,7,9]  GPU1:[6,8]

Tổng cộng 4 luồng GPU, phân bổ [3,2,3,2], mỗi run xuất hiện đúng một lần. Luồng
dài nhất là 3 run - tối ưu, vì ceil(10/4) = 3.

Vì sao KHÔNG dùng DataParallel/DDP:

1. Bài toán ở đây là 10 run ĐỘC LẬP (khác seed), tức song song hoàn hảo. Chia run
   giữa hai GPU cho tăng tốc gần tuyến tính mà không phải sửa một dòng nào trong
   vòng huấn luyện.
2. DataParallel chia batch 64 thành 32/32 rồi đồng bộ sau MỖI bước. Các mô hình ở
   đây nhỏ (hidden_dim=64), nên chi phí scatter/gather thường ăn hết phần lợi.
3. DataParallel làm thay đổi ngữ nghĩa batch và khiến việc cố định hạt giống phức
   tạp hơn. Cả dự án này đang đấu tranh cho tính tái lập (manifest, seed cố định);
   đánh đổi tính tái lập lấy vài phần trăm tốc độ là sai hướng.

Mỗi tiến trình chỉ nhìn thấy đúng một GPU qua CUDA_VISIBLE_DEVICES, nên code vẫn
dùng `cuda:0` như bình thường - không cần sửa run_experiments.py.

Lưu ý bộ nhớ: mỗi tiến trình tự dựng tensor cửa sổ trượt riêng
(Abilene 1.39 GB, Géant 1.14 GB, SDN 0.61 GB) cộng ma trận tabular và overhead
PyTorch. Hai tiến trình Abilene cần khoảng 5-6 GB RAM hệ thống. Kiểm tra RAM của
session trước khi tăng --procs lên quá 2.
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


def detect_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def split_runs(run_ids, n):
    """Chia vòng tròn để cân tải: run dài/ngắn trải đều giữa các tiến trình."""
    return [[r for i, r in enumerate(run_ids) if i % n == g] for g in range(n)]


def last_line(path, tail_bytes=8192):
    """Dòng không rỗng cuối cùng của file, đọc từ đuôi để không tốn bộ nhớ."""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            f.seek(max(0, size - tail_bytes))
            chunk = f.read().decode('utf-8', errors='replace')
    except Exception:
        return ''
    lines = [l.strip() for l in chunk.splitlines() if l.strip()]
    return lines[-1] if lines else ''


class AutoPusher:
    """Commit + push ngay khi có run mới hoàn tất.

    Vì sao: notebook trước đây chỉ push sau CẢ MỘT mô hình (5 run, ~8 giờ với
    Géant/STWaveFormer). Session bị ngắt ở giữa thì mọi run đã xong từ lần push
    trước đều mất, và lần sau phải train lại từ đầu.

    Dựa trên tín hiệu hoàn tất: manifest.json chỉ được ghi SAU khi run train và
    đánh giá xong (commit 815ccae). Chỉ tiến trình cha gọi git - hai tiến trình
    train không bao giờ đụng vào git, nên không có hai lệnh git tranh nhau
    .git/index.lock. Push thất bại không làm dừng training: commit vẫn nằm ở máy,
    lần sau thử lại.
    """

    def __init__(self, account, note_prefix):
        from tools.push_run import completed_run_dirs
        self._completed = completed_run_dirs
        self.account = account
        self.note_prefix = note_prefix
        # Run đã hoàn tất TRƯỚC khi bắt đầu (session trước, --skip_existing) thì
        # không cần push lại.
        self.pushed = set(completed_run_dirs())

    def poll(self, reason='auto'):
        new = sorted(set(self._completed()) - self.pushed)
        if not new:
            return
        names = ', '.join(d.split('/')[-2].split('_data_')[0] + '/' + d.split('/')[-1]
                          for d in new)
        print(f"  [PUSH] {len(new)} run mới hoàn tất ({names}) -> commit + push...", flush=True)
        r = subprocess.run(
            [sys.executable, 'tools/push_run.py', '--account', self.account,
             '--only_completed', '--note', f"{self.note_prefix} {reason}: {names}"],
            capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode == 0:
            self.pushed.update(new)
            print(f"  [PUSH] OK", flush=True)
        else:
            tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
            print(f"  [PUSH] THẤT BẠI (mã {r.returncode}) - training vẫn chạy tiếp, "
                  f"sẽ thử lại ở lần kiểm tra sau:", flush=True)
            for line in tail:
                print(f"         {line}", flush=True)


def monitor(running, every, pusher=None):
    """In tiến độ định kỳ cho tới khi mọi tiến trình kết thúc.

    stdout của từng tiến trình được ghi vào file riêng để hai luồng không trộn
    vào nhau. Nhưng nếu CHỈ ghi file thì trên Kaggle bạn ngồi nhìn màn hình trống
    hàng giờ, không biết tiến trình còn sống hay đã treo - và khi session bị ngắt
    cũng không biết nó dừng ở đâu. Hàm này in dòng mới nhất của mỗi GPU, có nhãn,
    và nếu có `pusher` thì push ngay các run vừa hoàn tất.

    `running`: danh sách (gpu_id, Popen, file_handle, run_ids, log_path).
    """
    last_shown = {}
    while any(p.poll() is None for _, p, _, _, _ in running):
        time.sleep(every)
        for g, _p, _f, _ids, log_path in running:
            line = last_line(log_path)
            if line and line != last_shown.get(g):
                last_shown[g] = line
                print(f"  [GPU{g}] {line}", flush=True)
        if pusher is not None:
            pusher.poll()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--run_ids', required=True, help="ví dụ '0,1,2,3,4'")
    ap.add_argument('--procs', type=int, default=None,
                    help="Số tiến trình. Mặc định = số GPU nhìn thấy được.")
    ap.add_argument('--log_dir', default='logs/_multigpu',
                    help="Nơi ghi stdout của từng tiến trình")
    ap.add_argument('--extra', default='',
                    help="Tham số truyền thẳng cho run_experiments.py. PHẢI viết dạng "
                         "có dấu bằng: --extra=\"--epochs 300 --patience 30\". Viết "
                         "--extra \"--epochs 300\" (dấu cách) thì argparse hiểu "
                         "'--epochs' là tuỳ chọn của chính script này và báo lỗi.")
    ap.add_argument('--progress_every', type=int, default=60,
                    help="Giây giữa hai lần in tiến độ. 0 = tắt, chỉ ghi file.")
    ap.add_argument('--auto_push', default=None, metavar='ACCOUNT',
                    help="Nhãn tài khoản (A/B). Bật thì mỗi run vừa hoàn tất được "
                         "commit + push ngay lên runs/<ACCOUNT>. Phải đang ở đúng "
                         "nhánh runs/<ACCOUNT> trước khi chạy.")
    a = ap.parse_args()

    run_ids = [int(x) for x in a.run_ids.split(',')]
    n_gpu = detect_gpus()
    procs = a.procs or max(1, n_gpu)

    if n_gpu == 0:
        print("Không thấy GPU nào - chạy một tiến trình trên CPU.")
        procs = 1
    elif procs > n_gpu:
        print(f"Yêu cầu {procs} tiến trình nhưng chỉ có {n_gpu} GPU. "
              f"Hai tiến trình dùng chung một GPU sẽ tranh VRAM và thường CHẬM HƠN "
              f"chạy tuần tự. Hạ xuống {n_gpu}.")
        procs = n_gpu

    groups = split_runs(run_ids, procs)
    os.makedirs(a.log_dir, exist_ok=True)

    print(f"GPU nhìn thấy: {n_gpu} | tiến trình: {procs}")
    for g, ids in enumerate(groups):
        print(f"  GPU {g}: run {ids}")
    print()

    running, t0 = [], time.time()
    for g, ids in enumerate(groups):
        if not ids:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = [sys.executable, '-u', 'run_experiments.py',
               '--dataset', a.dataset, '--model', a.model,
               '--run_ids', ','.join(map(str, ids)), '--skip_existing']
        if a.extra:
            cmd += a.extra.split()
        out_path = os.path.join(a.log_dir, f'{a.dataset}_{a.model}_gpu{g}.log')
        out = open(out_path, 'w', encoding='utf-8')
        print(f"  $ CUDA_VISIBLE_DEVICES={g} {' '.join(cmd)}")
        print(f"    -> {out_path}")
        running.append((g, subprocess.Popen(cmd, env=env, stdout=out,
                                            stderr=subprocess.STDOUT), out, ids, out_path))

    pusher = AutoPusher(a.auto_push, f"{a.dataset} {a.model}") if a.auto_push else None
    every = a.progress_every if a.progress_every > 0 else 60
    if running:
        print(f"\nTiến độ (cập nhật mỗi {every}s; log đầy đủ trong {a.log_dir}"
              f"{'; tự push từng run lên runs/' + a.auto_push if pusher else ''}):\n",
              flush=True)
        try:
            monitor(running, every, pusher)
        except KeyboardInterrupt:
            # Bạn bấm Interrupt: dừng các tiến trình train, nhưng TRƯỚC KHI thoát thì
            # push các run đã hoàn tất - để không phải nhớ chạy push_run bằng tay.
            print("\n[INTERRUPT] Dừng các tiến trình train...", flush=True)
            for _g, p, _f, _ids, _lp in running:
                if p.poll() is None:
                    p.terminate()
            for _g, p, _f, _ids, _lp in running:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
            if pusher is not None:
                pusher.poll(reason='interrupt')
            print("[INTERRUPT] Run đang dở không có manifest nên sẽ được train lại từ "
                  "đầu ở lần chạy sau (--skip_existing).", flush=True)
            return 130

    # Lần kiểm tra cuối: run hoàn tất giữa lần kiểm tra cuối cùng và lúc thoát.
    if pusher is not None:
        pusher.poll(reason='final')

    failed = []
    for g, p, out, ids, log_path in running:
        rc = p.wait()
        out.close()
        status = 'OK' if rc == 0 else f'LỖI (exit {rc})'
        print(f"  GPU {g} run {ids}: {status}")
        if rc != 0:
            failed.append((g, ids))
            # In đuôi log của tiến trình hỏng ngay tại đây: trên Kaggle, session
            # có thể kết thúc trước khi bạn kịp mở file log ra xem.
            print(f"    --- 15 dòng cuối {log_path} ---")
            try:
                with open(log_path, encoding='utf-8', errors='replace') as fh:
                    for line in fh.readlines()[-15:]:
                        print(f"    {line.rstrip()}")
            except Exception as e:
                print(f"    (không đọc được log: {e})")

    print(f"\nTổng thời gian: {(time.time() - t0) / 60:.1f} phút")
    if failed:
        print("CÓ TIẾN TRÌNH THẤT BẠI - xem log trong", a.log_dir)
        print("Các run còn thiếu sẽ được chạy lại nhờ --skip_existing ở lần sau.")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
