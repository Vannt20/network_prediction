import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import argparse
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines_ml.run_ml_baselines import run_ml_experiments
from training.precompute_cache import run_precompute
from training.train_gate_stacking import train_all_gates


def run_full_pipeline(datasets=None, models='all', runs=5, epochs=100, patience=20, quick_check=False, skip_existing=False, train_local=False):
    if datasets is None or 'all' in datasets:
        datasets = ['sdn', 'geant', 'abilene']

    actual_epochs = 2 if quick_check else epochs
    actual_patience = 2 if quick_check else patience

    print("=" * 90)
    print(" KHỞI ĐỘNG QUY TRÌNH THỰC NGHIỆM TOÀN DIỆN: ST-ADAPTIVE-ENSEMBLE")
    print(f" Datasets: {datasets} | Models: {models} | Runs: {runs} | Epochs: {actual_epochs} | Patience: {actual_patience}")
    print(f" Quick Check: {quick_check} | Skip Existing: {skip_existing} | Train Local: {train_local}")
    print("=" * 90)

    # -------------------------------------------------------------
    # BƯỚC 0 (TÙY CHỌN): HUẤN LUYỆN ĐỘC LẬP LOCALSPATIALTCN TRÊN GPU
    # -------------------------------------------------------------
    if train_local:
        print("\n[BƯỚC 0/5] HUẤN LUYỆN ĐỘC LẬP NHÁNH CỤC BỘ LOCALSPATIALTCN TRÊN GPU...")
        try:
            from run_experiments import run_all_experiments
            run_all_experiments(
                datasets=datasets,
                models=['LocalSpatialTCN'],
                epochs=actual_epochs,
                patience=actual_patience,
                runs=runs,
                skip_existing=skip_existing
            )
        except Exception as e:
            print(f"  [LỖI] Huấn luyện LocalSpatialTCN thất bại: {e}")
            raise e

    # -------------------------------------------------------------
    # BƯỚC 1: BENCHMARK MODULE A (TRADITIONAL ML & CHAMPION SELECTION)
    # -------------------------------------------------------------
    print("\n[BƯỚC 1/5] CHẠY MODULE A: BENCHMARK TRADITIONAL ML...")
    run_ml_experiments(
        models=models,
        datasets=datasets,
        runs=runs,
        quick_check=quick_check,
        skip_existing=skip_existing
    )

    # -------------------------------------------------------------
    # BƯỚC 2: TIỀN TÍNH TOÁN BỘ NHỚ ĐỆM (PRECOMPUTE CACHE)
    # -------------------------------------------------------------
    print("\n[BƯỚC 2/5] CHẠY OFFLINE PRECOMPUTE CACHE...")
    run_precompute(
        datasets=datasets,
        runs=runs,
        quick_check=quick_check
    )

    # -------------------------------------------------------------
    # BƯỚC 3: HUẤN LUYỆN CỔNG ĐIỀU PHỐI ĐỘNG (TRAIN STACKING GATE)
    # -------------------------------------------------------------
    print("\n[BƯỚC 3/5] HUẤN LUYỆN CỔNG PER-FLOW META-GATING TRÊN VALIDATION...")
    train_all_gates(
        datasets=datasets,
        runs=runs,
        epochs=20 if quick_check else 100
    )

    # -------------------------------------------------------------
    # BƯỚC 4: ABLATION STUDY
    # -------------------------------------------------------------
    print("\n[BƯỚC 4/5] ĐÁNH GIÁ 4 CẤU HÌNH ABLATION STUDY...")
    try:
        from evaluation.ablation_study import run_ablation_experiments
        run_ablation_experiments(datasets=datasets, runs=runs)
    except Exception as e:
        print(f"  (Cảnh báo Ablation: {e})")

    # -------------------------------------------------------------
    # BƯỚC 5: XUẤT BÁO CÁO VÀ BIỂU ĐỒ LUẬN VĂN
    # -------------------------------------------------------------
    print("\n[BƯỚC 5/5] XUẤT BÁO CÁO LUẬN VĂN & ĐỒ THỊ 300 DPI...")
    try:
        from evaluation.report_generator import generate_thesis_report
        from evaluation.plot_gate_dynamics import plot_all_thesis_figures
        generate_thesis_report()
        plot_all_thesis_figures()
    except Exception as e:
        print(f"  (Cảnh báo xuất báo cáo: {e})")

    print("\n" + "=" * 90)
    print(" TOÀN BỘ QUY TRÌNH THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG! ")
    print(" Kết quả tổng hợp có tại thư mục results/")
    print("=" * 90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Full ST-Adaptive-Ensemble Pipeline")
    parser.add_argument('--datasets', type=str, default='all', help="Comma-separated datasets hoặc 'all'")
    parser.add_argument('--models', type=str, default='all', help="Comma-separated models cho Module A hoặc 'all'")
    parser.add_argument('--runs', type=int, default=5, help="Số run lặp lại độc lập")
    parser.add_argument('--epochs', type=int, default=100, help="Số epoch tối đa cho huấn luyện Deep Learning (mặc định: 100)")
    parser.add_argument('--patience', type=int, default=20, help="Số epoch chờ Early Stopping (mặc định: 20)")
    parser.add_argument('--quick_check', action='store_true', help="Chạy kiểm tra nhanh toàn pipeline")
    parser.add_argument('--skip_existing', action='store_true', help="Bỏ qua các mô hình đã có kết quả")
    parser.add_argument('--train_local', action='store_true', help="Huấn luyện độc lập LocalSpatialTCN trên GPU trước khi Stacking")

    args = parser.parse_args()
    d_list = [d.strip() for d in args.datasets.split(',')] if args.datasets != 'all' else ['sdn', 'geant', 'abilene']
    m_list = [m.strip() for m in args.models.split(',')] if args.models != 'all' else 'all'

    run_full_pipeline(
        datasets=d_list,
        models=m_list,
        runs=args.runs,
        epochs=args.epochs,
        patience=args.patience,
        quick_check=args.quick_check,
        skip_existing=args.skip_existing,
        train_local=args.train_local
    )
