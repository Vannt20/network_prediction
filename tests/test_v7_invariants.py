"""Kiểm thử các bất biến của pipeline v7. Chạy: python -m pytest tests/ -q"""
import numpy as np
import pytest

from features.manifest import (Manifest, assert_compatible, PIPELINE_VERSION,
                               with_training_params, TRAINING_KEYS)
from Graph_models.pfar import PFAROffline, PFAROnline
from training.train_pfar import rolling_origin_blocks
from training.build_oof import oof_plan


def _mf(**kw):
    base = dict(pipeline_version=PIPELINE_VERSION, dataset='abilene', csv_sha256='a' * 8,
                n_steps_after_hygiene=47808, split_train=33465, split_val=4780,
                split_test=9563, seq_len=24, k_lags=12, n_windows_val=4756,
                n_windows_test=9539, seed=42, git_commit='x')
    base.update(kw)
    return Manifest(**base)


def test_manifest_phat_hien_lech():
    """I3 - lệch bất kỳ trường nghiêm ngặt nào phải RAISE, không được cảnh báo rồi đi tiếp."""
    assert_compatible(_mf(), _mf(seed=99))          # seed khác là hợp lệ
    with pytest.raises(RuntimeError):
        assert_compatible(_mf(), _mf(n_windows_test=9596))   # chính là lỗi B2


def test_manifest_bat_lech_sieu_tham_so_huan_luyen():
    """Hai run cùng dữ liệu nhưng khác --epochs KHÔNG phải cùng một thí nghiệm.

    Tình huống có thật: logs/localspatialtcn_data_sdn_seq_60/run_4 chạy 205 epoch
    trong khi trần của các run khác là 200. Trước khi bổ sung TRAINING_KEYS thì
    không cơ chế nào phát hiện được, và khi chia việc cho hai tài khoản Kaggle thì
    đây đúng là lỗi B2 trên một trục khác.
    """
    tp = dict(patience=30, warmup_epochs=15, lr=1e-3, weight_decay=1e-4)
    a = with_training_params(_mf(), epochs=200, **tp)
    b = with_training_params(_mf(), epochs=300, **tp)
    with pytest.raises(RuntimeError):
        assert_compatible(a, b)
    assert_compatible(a, with_training_params(_mf(), epochs=200, **tp))


def test_manifest_bo_qua_sieu_tham_so_khi_mot_ben_thieu():
    """Manifest của cache không mang siêu tham số huấn luyện -> phải bỏ qua, không raise.

    prepare_feature_store() sinh manifest trước khi biết epochs/patience, nên
    precompute_cache đối chiếu manifest dữ liệu với manifest run sẽ luôn gặp trường
    hợp một bên None. Đó là hợp lệ.
    """
    full = with_training_params(_mf(), epochs=200, patience=30, warmup_epochs=15,
                                lr=1e-3, weight_decay=1e-4)
    assert_compatible(_mf(), full)
    assert all(getattr(_mf(), k) is None for k in TRAINING_KEYS)


def test_pfar_khoi_phuc_he_so():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(300, 4, 3))
    y = 2.0 * P[..., 0] - 1.0 * P[..., 2] + 0.3
    W = PFAROffline(1e-10).fit(P, y).coef_
    assert np.allclose(W[0], [2.0, 0.0, -1.0, 0.3], atol=1e-4)


def test_pfar_online_nhan_qua():
    """Dự báo tại t chỉ được dùng thông tin đến t-1.

    Thay đổi nhãn ở đuôi chuỗi không được làm đổi bất kỳ dự báo nào trước đó.
    """
    rng = np.random.default_rng(1)
    Pf, yf = rng.normal(size=(50, 3, 3)), rng.normal(size=(50, 3))
    Pt, yt = rng.normal(size=(80, 3, 3)), rng.normal(size=(80, 3))
    a = PFAROnline(1e-4).run(Pf, yf, Pt, yt)
    yt2 = yt.copy()
    yt2[40:] += 100.0
    b = PFAROnline(1e-4).run(Pf, yf, Pt, yt2)
    assert np.allclose(a[:41], b[:41]), "rò rỉ nhân quả: nhãn tương lai ảnh hưởng dự báo quá khứ"
    assert not np.allclose(a[41:], b[41:]), "cập nhật trực tuyến không hoạt động"


def test_rolling_origin_khong_ro_ri():
    """I1 - khối đánh giá luôn nằm SAU khối khớp."""
    for f_idx, e_idx in rolling_origin_blocks(1000):
        assert f_idx[-1] < e_idx[0]


def test_oof_khong_ro_ri():
    """Fold dự đoán luôn nằm SAU fold huấn luyện."""
    plan = oof_plan(1000, 5)
    assert len(plan) == 4
    for _, tr, pr in plan:
        assert tr[-1] < pr[0]


def test_convex_bi_chan_boi_nhanh_manh_nhat():
    """Cơ sở toán học của việc bỏ ràng buộc lồi.

    Khi một nhánh vượt trội và sai số không phản tương quan, tổ hợp lồi tốt nhất
    chính là dồn 100% cho nhánh đó - tức hoà, không bao giờ thắng. Affine thì không
    bị chặn như vậy vì nó hiệu chỉnh được cả chệch lẫn tỷ lệ.
    """
    rng = np.random.default_rng(2)
    y = rng.normal(size=(500, 5))
    P = np.stack([y + rng.normal(0, 1.0, y.shape),
                  y + rng.normal(0, 1.0, y.shape),
                  1.2 * y + 0.4 + rng.normal(0, 0.1, y.shape)], -1)
    mse = lambda a: ((a - y) ** 2).mean()
    best_single = min(mse(P[..., k]) for k in range(3))
    affine = mse(PFAROffline(1e-8).fit(P, y).predict(P))
    assert affine < best_single
