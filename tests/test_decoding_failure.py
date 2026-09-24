"""Tests for the shared decoding-failure criterion."""
import numpy as np
import pytest

from astra_stim.sample_syndromes import bb_params
from codes import create_bivariate_bicycle_codes
from gnn_pipeline.decoding_failure import (
    component_failures,
    css_failures,
    css_failures_from_errors,
)


@pytest.fixture(scope="module")
def code72():
    css, _, _ = create_bivariate_bicycle_codes(*bb_params(6))
    return tuple(np.asarray(getattr(css, a)).astype(np.int64) for a in ("hx", "hz", "lx", "lz"))


def _random_errors(n, shots, p, seed):
    rng = np.random.default_rng(seed)
    return (rng.random((shots, n)) < p).astype(np.int64), (rng.random((shots, n)) < p / 21).astype(np.int64)


def test_exact_decoding_is_success(code72):
    hx, hz, lx, lz = code72
    z, x = _random_errors(hx.shape[1], 50, 0.04, 0)
    report = css_failures_from_errors(z, x, z, x, hx, hz, lx, lz)
    assert not report.failure.any()


def test_stabilizer_residual_is_success(code72):
    hx, hz, lx, lz = code72
    z, x = _random_errors(hx.shape[1], 1, 0.04, 1)
    z_hat = (z + hz[3]) % 2
    x_hat = (x + hx[5]) % 2
    report = css_failures_from_errors(z_hat, x_hat, z, x, hx, hz, lx, lz)
    assert not report.failure.any()


def test_logical_residual_is_a_logical_flip(code72):
    hx, hz, lx, lz = code72
    i = next(i for i in range(lz.shape[0]) if ((lx @ lz[i]) % 2).any())
    z = np.zeros((1, hx.shape[1]), dtype=np.int64)
    report = css_failures_from_errors(lz[i][None, :], z, z, z, hx, hz, lx, lz)
    assert report.logical_flip.all() and not report.syndrome_mismatch.any()


def test_syndrome_mismatch_fails_even_when_observables_agree(code72):
    hx, hz, lx, lz = code72
    n = hx.shape[1]
    candidates = [[q] for q in range(n)] + [[q, q2] for q in range(n) for q2 in range(q + 1, n)]
    residual = None
    for support in candidates:
        r = np.zeros(n, dtype=np.int64)
        r[support] = 1
        if not ((lx @ r) % 2).any() and ((hx @ r) % 2).any():
            residual = r
            break
    assert residual is not None
    zero = np.zeros((1, n), dtype=np.int64)
    report = css_failures_from_errors(zero, zero, residual[None, :], zero, hx, hz, lx, lz)
    assert report.syndrome_mismatch.all()
    assert report.failure.all()


def test_simultaneous_x_and_z_logical_flips_do_not_cancel(code72):
    hx, hz, lx, lz = code72
    i = next(i for i in range(lx.shape[0]) if (lx[i] @ lz[i]) % 2)
    zero = np.zeros((1, hx.shape[1]), dtype=np.int64)
    report = css_failures_from_errors(lz[i][None, :], lx[i][None, :], zero, zero, hx, hz, lx, lz)
    assert report.failure.all()


def test_categories_are_disjoint_and_or_combines_components(code72):
    hx, hz, lx, lz = code72
    n = hx.shape[1]
    z, x = _random_errors(n, 400, 0.08, 2)
    rng = np.random.default_rng(3)
    z_hat = (z + (rng.random(z.shape) < 0.02)) % 2
    x_hat = (x + (rng.random(x.shape) < 0.02)) % 2
    report = css_failures_from_errors(z_hat, x_hat, z, x, hx, hz, lx, lz)
    assert not (report.syndrome_mismatch & report.logical_flip).any()
    obs = np.concatenate([(z @ lx.T) % 2, (x @ lz.T) % 2], axis=1)
    zp = component_failures(z_hat, (z @ hx.T) % 2, hx, lx, obs[:, :lx.shape[0]])
    xp = component_failures(x_hat, (x @ hz.T) % 2, hz, lz, obs[:, lx.shape[0]:])
    np.testing.assert_array_equal(report.failure, zp.failure | xp.failure)


def test_stored_observables_match_derived(code72):
    hx, hz, lx, lz = code72
    z, x = _random_errors(hx.shape[1], 100, 0.05, 4)
    z_hat = np.zeros_like(z)
    obs = np.concatenate([(z @ lx.T) % 2, (x @ lz.T) % 2], axis=1)
    a = css_failures(z_hat, x, (z @ hx.T) % 2, (x @ hz.T) % 2, hx, hz, lx, lz, obs.astype(np.float32))
    b = css_failures_from_errors(z_hat, x, z, x, hx, hz, lx, lz)
    np.testing.assert_array_equal(a.failure, b.failure)


def test_single_shot_inputs(code72):
    hx, hz, lx, lz = code72
    z, x = _random_errors(hx.shape[1], 1, 0.04, 5)
    report = css_failures_from_errors(z[0], x[0], z[0], x[0], hx, hz, lx, lz)
    assert report.failure.shape == (1,) and not report.failure[0]


def test_observable_width_mismatch_raises(code72):
    hx, hz, lx, lz = code72
    z, x = _random_errors(hx.shape[1], 2, 0.04, 6)
    obs = np.zeros((2, lx.shape[0]), dtype=np.int64)
    with pytest.raises(ValueError):
        css_failures(z, x, (z @ hx.T) % 2, (x @ hz.T) % 2, hx, hz, lx, lz, obs)


def test_soft_values_rejected(code72):
    hx, hz, lx, lz = code72
    n = hx.shape[1]
    with pytest.raises(ValueError):
        component_failures(np.full((1, n), 0.3), np.zeros((1, hx.shape[0])), hx, lx, np.zeros((1, lx.shape[0])))


def test_evaluate_wrappers_use_shared_definition(code72):
    from gnn_pipeline.evaluate import _check_logical_error, _check_logical_errors_batch

    hx, hz, lx, lz = code72
    z, x = _random_errors(hx.shape[1], 200, 0.08, 7)
    rng = np.random.default_rng(8)
    z_hat = (z + (rng.random(z.shape) < 0.02)) % 2
    x_syn, z_syn = (z @ hx.T) % 2, (x @ hz.T) % 2
    obs = np.concatenate([(z @ lx.T) % 2, (x @ lz.T) % 2], axis=1)
    expected = css_failures(z_hat, x, x_syn, z_syn, hx, hz, lx, lz, obs).failure
    batch = _check_logical_errors_batch(z_hat, x, x_syn, z_syn, hx, hz, lx, lz, obs)
    single = [_check_logical_error(z_hat[i], x[i], x_syn[i], z_syn[i], hx, hz, lx, lz, obs[i]) for i in range(200)]
    np.testing.assert_array_equal(batch, expected)
    np.testing.assert_array_equal(single, expected)
    assert expected.any()
