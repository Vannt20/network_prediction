"""
Ablation 10 cấu hình cho v7. Thay evaluation/ablation_study.py (giữ lại file cũ).

A3 và A10 là hai dòng quan trọng nhất VÀ nguy hiểm nhất cho luận văn: chúng đo
xem bao nhiêu phần cải thiện chỉ đến từ việc hiệu chỉnh chệch/tỷ lệ của MỘT nhánh,
và hai nhánh học sâu thật sự đóng góp bao nhiêu.

Số đã đo trên SDN: A2 = 5.509, A3 = 4.494, A9 = 4.088. Dưới bộ gộp phi tuyến, phân
rã đóng góp là: chỉ ML 4.181 -> +Global 4.088 -> +Local 4.087 -> cả ba 4.076. Nghĩa
là hai nhánh học sâu cộng lại chỉ thêm ~2.5%.

Phải chủ động trình bày con số này. Giấu A3/A10 rồi chỉ so A9 với A5 để khoe mức
giảm 38% là cách chắc chắn bị hội đồng bắt lỗi.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Graph_models.pfar import PFAROffline, PFAROnline
from Graph_models.stacker import TwoStageCombiner
from evaluation.baselines_naive import skill_score

CONFIGS = [
    ('A1',  'persistence',           'Bài toán có tầm thường không?'),
    ('A2',  'best_single',           'Gộp có cần thiết không?'),
    ('A3',  'best_single_recalib',   'Bao nhiêu phần cải thiện chỉ từ hiệu chỉnh affine?'),
    ('A4',  'static_average',        'Gộp ngây thơ 1/3 đã đủ chưa?'),
    ('A5',  'softmax_gate',          'Mô hình cũ đứng ở đâu?'),
    ('A6',  'pfar_offline_val',      'Đóng góp của việc bỏ ràng buộc lồi'),
    ('A7',  'pfar_offline_oof',      'Đóng góp của ma trận OOF'),
    ('A8',  'pfar_online',           'Đóng góp của hiệu chỉnh trực tuyến'),
    ('A9',  'two_stage_online',      'Đóng góp của tầng gộp phi tuyến (S2)'),
    ('A10', 'two_stage_ml_only',     'Hai nhánh học sâu thật sự đóng góp bao nhiêu?'),
]

_mse = lambda a, y: float(((np.asarray(a, np.float64) - np.asarray(y, np.float64)) ** 2).mean())


def run_ablation(Pv, xv, cv, yv, Pt, xt, ct, yt, hp, P_oof=None, x_oof=None,
                 c_oof=None, y_oof=None, softmax_pred=None):
    """Chạy đủ 10 cấu hình cho MỘT run. Trả dict {mã: MSE}.

    P_* có thứ tự cột [Global, Local, ML]. hp là siêu tham số đã chọn bằng
    rolling-origin trên khối Train+Val (không bao giờ trên Test).
    """
    ridge, lam, anchor = hp['ridge'], hp['lam'], hp['anchor']
    out = {}

    out['A1'] = _mse(xt, yt)

    # Nhánh mạnh nhất xác định trên VAL, không phải trên Test.
    k_best = int(np.argmin([_mse(Pv[..., k], yv) for k in range(Pv.shape[-1])]))
    out['A2'] = _mse(Pt[..., k_best], yt)

    single_v = Pv[..., [k_best]]
    single_t = Pt[..., [k_best]]
    out['A3'] = _mse(PFAROffline(ridge).fit(single_v, yv).predict(single_t), yt)

    out['A4'] = _mse(Pt.mean(axis=-1), yt)
    out['A5'] = _mse(softmax_pred, yt) if softmax_pred is not None else float('nan')
    out['A6'] = _mse(PFAROffline(ridge).fit(Pv, yv).predict(Pt), yt)

    if P_oof is not None:
        out['A7'] = _mse(PFAROffline(ridge).fit(P_oof, y_oof).predict(Pt), yt)
    else:
        out['A7'] = float('nan')   # Stage B

    out['A8'] = _mse(PFAROnline(ridge, lam, anchor).run(Pv, yv, Pt, yt), yt)

    # Khối kiểm định cho cổng C7: nửa sau của khối khớp. Không bao giờ là Test.
    if P_oof is not None:
        Pf, xf, cf, yf = P_oof, x_oof, c_oof, y_oof
    else:
        Pf, xf, cf, yf = Pv, xv, cv, yv
    h = len(Pf) // 2
    comb = TwoStageCombiner(ridge, lam, anchor).fit(
        Pf[:h], xf[:h], cf[:h], yf[:h], Pf[h:], xf[h:], cf[h:], yf[h:])
    out['A9'] = _mse(comb.predict_online(Pf, xf, cf, yf, Pt, xt, ct, yt), yt)
    out['_c7_kept'] = bool(comb.c7_passed)

    # A10: bỏ hẳn hai nhánh học sâu khỏi bộ gộp.
    ml = [Pv.shape[-1] - 1]
    comb_ml = TwoStageCombiner(ridge, lam, anchor).fit(
        Pf[:h][..., ml], xf[:h], cf[:h], yf[:h], Pf[h:][..., ml], xf[h:], cf[h:], yf[h:])
    out['A10'] = _mse(comb_ml.predict_online(
        Pf[..., ml], xf, cf, yf, Pt[..., ml], xt, ct, yt), yt)

    return out


def summarize(rows, ds_key):
    """In bảng ablation kèm skill score so với persistence."""
    keys = [c[0] for c in CONFIGS]
    pers = float(np.mean([r['A1'] for r in rows]))
    print(f"\n{'=' * 84}\nABLATION {ds_key.upper()}  ({len(rows)} run)\n{'=' * 84}")
    print(f"{'#':<5}{'cấu hình':<24}{'MSE x1e-3':>16}{'skill':>9}{'câu hỏi':<30}")
    print('-' * 84)
    table = {}
    for code, name, question in CONFIGS:
        vals = np.array([r[code] for r in rows], dtype=np.float64)
        m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
        table[code] = m
        mstr = 'n/a (Stage B)' if np.isnan(m) else f'{m * 1e3:8.3f}±{s * 1e3:<5.3f}'
        sk = '' if np.isnan(m) else f'{skill_score(m, pers):>9.3f}'
        print(f"{code:<5}{name:<24}{mstr:>16}{sk}  {question}")
    if not np.isnan(table['A9']) and not np.isnan(table['A10']):
        gain = 100.0 * (table['A10'] - table['A9']) / table['A10']
        print(f"\n  Đóng góp thực của hai nhánh học sâu (A10 -> A9): {gain:.1f}%")
    return table
