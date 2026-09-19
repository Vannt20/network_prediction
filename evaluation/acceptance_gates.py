"""
Cổng nghiệm thu C1-C7. Chạy tự động sau mỗi run, ghi logs/<model>/<run>/gates.json.

C1 và C2 là hai cổng quan trọng nhất: nếu đã có từ đầu thì lỗi B1 (nhánh Local sụp
về persistence trên Géant và Abilene, train_loss tăng đều từ epoch 1 đến 70, best
checkpoint chính là trạng thái khởi tạo) đã bị bắt ngay ngày đầu tiên thay vì phát
hiện ở giai đoạn viết luận văn.

Nguyên tắc I6: run nào không qua đủ cổng thì KHÔNG được đưa vào bộ gộp.
"""
import os
import json
import numpy as np
import pandas as pd


def c1_branch_alive(y_pred, x_last, mse_persistence, thresh=0.05):
    """Nhánh còn sống: đầu ra phải lệch khỏi persistence một lượng đáng kể.

    Bắt đúng lỗi B1: trên Géant và Abilene, mean((y_local - x_{t-1})^2) = 0.000,
    tức nhánh Local là persistence trá hình.
    """
    dev = float(((np.asarray(y_pred, np.float64) - np.asarray(x_last, np.float64)) ** 2).mean())
    return dev > thresh * mse_persistence, {'deviation': dev,
                                            'threshold': thresh * mse_persistence}


def c2_actually_learned(train_metrics_csv, warmup_epochs=15, ratio=0.95):
    """Có học thật: train_loss cuối phải thấp hơn train_loss tại mốc warmup.

    Bắt trường hợp early stopping giữ lại đúng trạng thái khởi tạo.
    """
    if not os.path.exists(train_metrics_csv):
        return False, {'error': f'không có {train_metrics_csv}'}
    df = pd.read_csv(train_metrics_csv)
    if len(df) <= warmup_epochs:
        return False, {'error': f'chỉ có {len(df)} epoch, ít hơn warmup={warmup_epochs}'}
    ref = float(df['train_loss'].iloc[min(warmup_epochs, len(df)) - 1])
    last = float(df['train_loss'].iloc[-1])
    return last < ratio * ref, {'train_loss_at_warmup': ref, 'train_loss_final': last}


def c3_beats_persistence(mse_branch, mse_persistence):
    return mse_branch < mse_persistence, {'mse_branch': mse_branch,
                                          'mse_persistence': mse_persistence}


def c4_loss_matches_metric(loss_name):
    """Hàm mất mát phải cùng họ với chỉ số công bố (MSE).

    Bắt lỗi B3: huấn luyện bằng SmoothL1Loss(beta=0.01) / huber_loss(delta=0.01)
    rồi công bố MSE. Trên thang MinMax [0,1], MAE điển hình của các nhánh là
    0.017-0.067 nên gần như mọi mẫu rơi vào vùng tuyến tính -> đang tối ưu MAE.
    """
    ok = str(loss_name).lower().replace('()', '') in ('mseloss', 'mse', 'l2')
    return ok, {'loss': str(loss_name)}


def c5_cache_consistent(cache_dir, ds_key):
    """Mọi cache cùng dataset phải có manifest tương thích. Bắt lỗi B2."""
    import glob
    import torch
    from features.manifest import assert_compatible
    files = sorted(glob.glob(os.path.join(cache_dir, f'{ds_key}_run_*_test_preds.pt')))
    if not files:
        return False, {'error': 'không có cache nào'}
    ref = None
    for f in files:
        d = torch.load(f, weights_only=False)
        if 'manifest' not in d:
            return False, {'error': f'{os.path.basename(f)} không có manifest (cache pipeline cũ)'}
        if ref is None:
            ref = d['manifest']
        else:
            try:
                assert_compatible(ref, d['manifest'])
            except RuntimeError as e:
                return False, {'error': str(e), 'file': os.path.basename(f)}
    return True, {'n_files': len(files), 'n_windows_test': ref.get('n_windows_test')}


def c6_no_leakage(es_index, combiner_index):
    """Tập early stopping của nhánh ML và tập khớp bộ gộp phải rời nhau (I5).

    Bắt lỗi B4: XGBoost early-stop trên chính Validation mà cổng dùng để học trọng
    số. Hậu quả trên Abilene: nhánh ML tốt nhất trên Val (0.961) nhưng tệ nhất trên
    Test (2.763), bộ gộp dồn 79% trọng số cho nó.
    """
    inter = set(np.asarray(es_index).tolist()) & set(np.asarray(combiner_index).tolist())
    return len(inter) == 0, {'n_overlap': len(inter)}


def c7_nonlinear_justified(c7_passed, detail):
    """Tầng phi tuyến phải tự chứng minh trên khối kiểm định, nếu không thì rút về
    PFAR affine. Đây là cổng của sửa đổi S2 - ngăn việc ép một bộ gộp cho cả ba tập
    khi trên Géant tầng phi tuyến khớp trên Val làm MSE xấu đi từ 0.815 lên 1.229.

    Cổng này KHÔNG bao giờ làm dừng pipeline: nó chỉ ghi lại quyết định giữ/bỏ.
    """
    return True, {'nonlinear_kept': bool(c7_passed), **(detail or {})}


def c8_beats_every_branch(mse_combiner, mse_branches, mse_persistence=None):
    """Bộ gộp phải tốt hơn MỌI nhánh đơn lẻ (và tốt hơn persistence).

    Đây là yêu cầu cứng: một bộ gộp thua nhánh đơn lẻ tốt nhất thì không có lý do
    tồn tại, và hội đồng sẽ hỏi đúng câu đó.

    Cổng này đo trên tập Test nên KHÔNG được dùng để chọn tham số (vi phạm I1) -
    nó chỉ để BÁO CÁO. Bảo đảm thật nằm ở chỗ khác, trong cấu trúc của bộ gộp:
    PFAR co hệ số về one-hot(nhánh tốt nhất) thay vì về 0, nên khi ridge đủ lớn nó
    hội tụ đúng về nhánh đơn lẻ đó. Nhánh đơn lẻ tốt nhất vì thế là một ứng viên
    trong lưới mà rolling-origin đang dò, và bộ gộp không thể thua nó một cách có
    hệ thống. Xem Graph_models/pfar.py::best_single_prior.
    """
    worst_gap = min(float(b) - float(mse_combiner) for b in mse_branches)
    ok = worst_gap > 0
    detail = {'mse_combiner': float(mse_combiner),
              'mse_branches': [float(b) for b in mse_branches],
              'margin_vs_best_branch': worst_gap}
    if mse_persistence is not None:
        detail['mse_persistence'] = float(mse_persistence)
        detail['skill_persist'] = 1.0 - float(mse_combiner) / float(mse_persistence)
        ok = ok and mse_combiner < mse_persistence
    return ok, detail


def run_all_gates(logdir, y_pred, x_last, mse_persistence, loss_name,
                  warmup_epochs=15, cache_dir=None, ds_key=None,
                  es_index=None, combiner_index=None, c7=None, c8=None, write=True):
    """Chạy toàn bộ cổng áp dụng được. Trả dict kèm khoá 'passed'.

    passed=False -> pipeline phải DỪNG, không được chạy tiếp sang bộ gộp (I6).
    """
    mse_branch = float(((np.asarray(y_pred, np.float64) - np.asarray(x_last, np.float64)) ** 2).mean()) \
        if x_last is None else None
    res, det = {}, {}

    res['C1'], det['C1'] = c1_branch_alive(y_pred, x_last, mse_persistence)
    res['C2'], det['C2'] = c2_actually_learned(os.path.join(logdir, 'train_metrics.csv'), warmup_epochs)
    res['C4'], det['C4'] = c4_loss_matches_metric(loss_name)
    if cache_dir and ds_key:
        res['C5'], det['C5'] = c5_cache_consistent(cache_dir, ds_key)
    if es_index is not None and combiner_index is not None:
        res['C6'], det['C6'] = c6_no_leakage(es_index, combiner_index)
    if c7 is not None:
        res['C7'], det['C7'] = c7_nonlinear_justified(c7.get('passed'), c7.get('detail'))
    if c8 is not None:
        res['C8'], det['C8'] = c8_beats_every_branch(
            c8['mse_combiner'], c8['mse_branches'], c8.get('mse_persistence'))

    out = {**res, 'passed': all(res.values()), 'details': det}
    if write:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, 'gates.json'), 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    return out


def add_c3(gates, mse_branch, mse_persistence, logdir=None):
    """C3 cần MSE trên Test nên được bổ sung sau, khi đã có dự đoán Test."""
    ok, d = c3_beats_persistence(mse_branch, mse_persistence)
    gates['C3'] = ok
    gates['details']['C3'] = d
    gates['passed'] = all(v for k, v in gates.items() if k.startswith('C'))
    if logdir:
        with open(os.path.join(logdir, 'gates.json'), 'w', encoding='utf-8') as f:
            json.dump(gates, f, indent=2, ensure_ascii=False, default=str)
    return gates
