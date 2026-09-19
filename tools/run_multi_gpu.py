"""
Chạy nhiều run song song, mỗi GPU một tiến trình. Dành cho Kaggle "GPU T4 x2".

    python tools/run_multi_gpu.py --dataset sdn --model STWaveFormer --run_ids 0,1,2,3,4

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
                    help="Tham số truyền thẳng cho run_experiments.py, "
                         "ví dụ \"--epochs 300 --patience 30\"")
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
                                            stderr=subprocess.STDOUT), out, ids))

    failed = []
    for g, p, out, ids in running:
        rc = p.wait()
        out.close()
        status = 'OK' if rc == 0 else f'LỖI (exit {rc})'
        print(f"  GPU {g} run {ids}: {status}")
        if rc != 0:
            failed.append((g, ids))

    print(f"\nTổng thời gian: {(time.time() - t0) / 60:.1f} phút")
    if failed:
        print("CÓ TIẾN TRÌNH THẤT BẠI - xem log trong", a.log_dir)
        print("Các run còn thiếu sẽ được chạy lại nhờ --skip_existing ở lần sau.")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
