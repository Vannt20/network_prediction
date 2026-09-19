"""
Xác thực bộ gộp v7 trên cache SẴN CÓ - không GPU, không huấn luyện lại, < 10 phút.

Mục đích: chứng minh hướng đi đúng TRƯỚC khi cam kết hàng chục giờ GPU. Nếu script
này không tái lập được số tham chiếu thì có gì đó sai trong cài đặt và không được
chạy tiếp pipeline.

    python tools/quick_validate_v7.py                 # cả ba tập
    python tools/quick_validate_v7.py --datasets sdn  # một tập
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import MinMaxScaler
from features.feature_store import load_raw_dataset
from Graph_models.pfar import PFAROffline, PFAROnline
from Graph_models.stacker import TwoStageCombiner
from evaluation.baselines_naive import skill_score

# Ridge/lambda đã dò. CHỈ dùng cho kiểm thử hồi quy của script này.
# Trong pipeline thật, chúng phải được chọn bằng rolling-origin trên Train+Val
# (training/train_pfar.py). Không được hard-code vào pipeline.
REF_RIDGE = {'sdn': 1e-5, 'geant': 1e-2, 'abilene': 1e-3}

# Run hợp lệ: Géant/Abilene run 0-4 chứa artifact hỏng B2 (Global MSE 10.37 thay vì
# 1.98 trên Abilene) nên bị loại khỏi kiểm chứng.
VALID_RUNS = {'sdn': list(range(10)), 'geant': list(range(5, 10)), 'abilene': list(range(5, 10))}

# MSE x1e-3 trên Test, dung sai +-3%.
REFERENCE = {
    'sdn':     {'persistence': 26.554, 'global': 15.350, 'local': 15.152, 'ml': 5.509,
                'oracle_convex': 5.509, 'pfar_offline': 4.403, 'pfar_online': 4.312},
    'geant':   {'persistence': 0.855, 'global': 0.738, 'local': 0.854, 'ml': 0.916,
                'oracle_convex': 0.727, 'pfar_offline': 1.005, 'pfar_online': 0.815},
    'abilene': {'persistence': 2.363, 'global': 1.978, 'local': 2.361, 'ml': 2.764,
                'oracle_convex': 1.905, 'pfar_offline': 2.280, 'pfar_online': 1.620},
}

MSE = lambda a, y: float(((np.asarray(a, np.float64) - np.asarray(y, np.float64)) ** 2).mean() * 1e3)


def load_cache(ds, run, split):
    path = os.path.join('cache', f'{ds}_run_{run}_{split}_preds.pt')
    d = torch.load(path, weights_only=False)
    P = np.stack([d['y_global'].numpy(), d['y_local'].numpy(), d['y_ml'].numpy()], -1).astype(np.float64)
    return P, d['y_real'].numpy().astype(np.float64), d['context'].numpy().astype(np.float32)


def reconstruct_x_last(ds, y_val, y_test):
    """Dựng lại giá trị bước t-1 (persistence) bằng cách căn chỉnh cache với chuỗi gốc.

    Cache hiện tại chưa lưu `x_last`; pipeline v7 sẽ lưu sẵn trường này
    (training/precompute_cache.py) nên bước căn chỉnh này chỉ cần cho cache cũ.
    """
    df = load_raw_dataset(ds)
    T = len(df)
    n_train = int(T * 0.7)
    full = MinMaxScaler().fit(df.iloc[:n_train]).transform(df).astype(np.float64)

    def find(Y):
        M = len(Y)
        for off in range(1, T - M + 1):
            if np.abs(full[off:off + M] - Y).max() < 1e-4:
                return full[off - 1:off - 1 + M]
        raise RuntimeError(f"Không căn chỉnh được cache {ds} với chuỗi gốc.")

    return find(y_val), find(y_test)


def oracle_convex(P, Y):
    """Tổ hợp lồi TỐT NHẤT CÓ THỂ trên chính tập Test (nghiệm chính xác, không phải
    gradient descent): liệt kê đủ các mặt của đơn hình và giải KKT trên từng mặt.

    Đây là trần của mọi cơ chế softmax gating. Trên SDN nghiệm là w = [0, 0, 1].
    """
    K = P.shape[-1]
    X = P.reshape(-1, K)
    y = Y.reshape(-1)
    A = X.T @ X / len(y)
    b = X.T @ y / len(y)
    best = (np.inf, None)
    for mask in range(1, 1 << K):
        idx = [i for i in range(K) if mask >> i & 1]
        n = len(idx)
        KKT = np.zeros((n + 1, n + 1))
        KKT[:n, :n] = A[np.ix_(idx, idx)]
        KKT[:n, n] = 1.0
        KKT[n, :n] = 1.0
        try:
            sol = np.linalg.solve(KKT, np.concatenate([b[idx], [1.0]]))[:n]
        except np.linalg.LinAlgError:
            continue
        if (sol < -1e-9).any():
            continue
        w = np.zeros(K)
        w[idx] = sol
        m = float(((X @ w - y) ** 2).mean())
        if m < best[0]:
            best = (m, w)
    return best[0] * 1e3, best[1]


def run_dataset(ds):
    runs = VALID_RUNS[ds]
    ridge = REF_RIDGE[ds]
    acc = {k: [] for k in ['persistence', 'global', 'local', 'ml', 'oracle_convex',
                           'oracle_select', 'pfar_offline', 'pfar_online',
                           'two_stage_static', 'two_stage_online']}
    c7 = []
    for r in runs:
        Pv, yv, cv = load_cache(ds, r, 'val')
        Pt, yt, ct = load_cache(ds, r, 'test')
        xv, xt = reconstruct_x_last(ds, yv, yt)

        acc['persistence'].append(MSE(xt, yt))
        for i, name in enumerate(['global', 'local', 'ml']):
            acc[name].append(MSE(Pt[..., i], yt))
        acc['oracle_convex'].append(oracle_convex(Pt, yt)[0])
        acc['oracle_select'].append(float(((Pt - yt[..., None]) ** 2).min(-1).mean() * 1e3))

        acc['pfar_offline'].append(MSE(PFAROffline(ridge).fit(Pv, yv).predict(Pt), yt))
        acc['pfar_online'].append(MSE(PFAROnline(ridge).run(Pv, yv, Pt, yt), yt))

        # S2: bộ gộp hai tầng. Khớp trên Val (chưa có OOF), khối kiểm định là nửa
        # sau của Val -> cổng C7 tự quyết định giữ hay bỏ tầng phi tuyến.
        h = len(Pv) // 2
        comb = TwoStageCombiner(ridge=ridge).fit(
            Pv[:h], xv[:h], cv[:h], yv[:h], Pv[h:], xv[h:], cv[h:], yv[h:])
        c7.append(comb.c7_passed)
        acc['two_stage_static'].append(MSE(comb.predict_static(Pv, xv, cv, yv, Pt, xt, ct), yt))
        acc['two_stage_online'].append(MSE(comb.predict_online(Pv, xv, cv, yv, Pt, xt, ct, yt), yt))

    pers = float(np.mean(acc['persistence']))
    print(f"\n{'=' * 78}\n{ds.upper()}  (run hợp lệ: {runs}, ridge={ridge:g})\n{'=' * 78}")
    print(f"{'cấu hình':<24}{'MSE x1e-3':>14}{'skill':>9}{'tham chiếu':>13}{'':>8}")
    print('-' * 78)
    ok_all = True
    for k, v in acc.items():
        m, s = float(np.mean(v)), float(np.std(v))
        ref = REFERENCE[ds].get(k)
        if ref is None:
            flag = ''
        elif abs(m - ref) <= 0.03 * ref:
            flag = 'PASS'
        else:
            flag = 'FAIL'
            ok_all = False
        refs = f'{ref:.3f}' if ref is not None else '-'
        print(f"{k:<24}{m:>9.3f}±{s:<4.3f}{skill_score(m, pers):>9.3f}{refs:>13}{flag:>8}")
    print(f"\nCổng C7 (giữ tầng phi tuyến): {sum(c7)}/{len(c7)} run")
    return ok_all


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='sdn,geant,abilene')
    a = ap.parse_args()
    results = {d: run_dataset(d) for d in [x.strip() for x in a.datasets.split(',')]}
    print(f"\n{'=' * 78}")
    for d, ok in results.items():
        print(f"  {d:<10} {'PASS' if ok else 'FAIL - kiểm tra lại cài đặt trước khi chạy pipeline'}")
    sys.exit(0 if all(results.values()) else 1)
