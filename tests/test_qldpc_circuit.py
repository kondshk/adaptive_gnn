"""Tests for the repeated-measurement memory circuit and its detectors."""
import numpy as np
import pytest
import stim

from astra_stim.biased_noise import BiasedCircuitNoiseSpec, apply_biased_circuit_noise
from astra_stim.qldpc_circuit import CSSCodeSpec, build_qldpc_memory_circuit_text
from astra_stim.sample_syndromes import bb_params
from codes import create_bivariate_bicycle_codes


@pytest.fixture(scope="module")
def spec72():
    css, _, _ = create_bivariate_bicycle_codes(*bb_params(6))
    return CSSCodeSpec(*(np.asarray(getattr(css, a), dtype=np.uint8) for a in ("hx", "hz", "lx", "lz")))


def _circuit(spec, rounds, basis, p=0.0):
    c = stim.Circuit(build_qldpc_memory_circuit_text(spec, rounds=rounds, basis=basis))
    if p:
        c = apply_biased_circuit_noise(c, spec=BiasedCircuitNoiseSpec(p=p, eta=20.0))
    return c


@pytest.mark.parametrize("basis", ["z", "x"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_detector_and_observable_counts(spec72, basis, rounds):
    mx, mz, k = spec72.hx.shape[0], spec72.hz.shape[0], spec72.lx.shape[0]
    c = _circuit(spec72, rounds, basis)
    boundary = mz if basis == "z" else mx
    assert c.num_detectors == (rounds - 1) * (mx + mz) + 2 * boundary
    assert c.num_observables == k


@pytest.mark.parametrize("basis", ["z", "x"])
def test_noiseless_circuit_is_deterministic(spec72, basis):
    c = _circuit(spec72, 3, basis)
    dets, obs = c.compile_detector_sampler(seed=0).sample(64, separate_observables=True)
    assert not dets.any() and not obs.any()


@pytest.mark.parametrize("basis", ["z", "x"])
def test_every_logical_fault_is_detected(spec72, basis):
    """No single fault may flip an observable without firing a detector."""
    dem = _circuit(spec72, 3, basis, p=1e-3).detector_error_model(approximate_disjoint_errors=True)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        targets = inst.targets_copy()
        if any(t.is_logical_observable_id() for t in targets):
            assert any(t.is_relative_detector_id() for t in targets), str(inst)


def test_final_data_error_is_detected(spec72):
    """A Z-basis data flip after the last round must fire the final detectors."""
    base = build_qldpc_memory_circuit_text(spec72, rounds=2, basis="z")
    head, tail = base.rsplit("\nM ", 1)
    c = stim.Circuit(head + "\nX_ERROR(1) 0\nM " + tail)
    dets = c.compile_detector_sampler(seed=0).sample(1)[0]
    mz = spec72.hz.shape[0]
    fired = np.flatnonzero(dets)
    assert fired.size == int(spec72.hz[:, 0].sum())
    assert (fired >= c.num_detectors - mz).all()


def test_rescale_noise_per_qubit(spec72):
    from astra_stim.biased_noise import rescale_noise_per_qubit

    c = _circuit(spec72, 2, "z", p=1e-3)
    scale = np.linspace(0.5, 3.0, c.num_qubits)
    r = rescale_noise_per_qubit(c, scale)
    assert r.num_detectors == c.num_detectors and r.num_observables == c.num_observables
    for op in r:
        if op.name in {"PAULI_CHANNEL_1", "X_ERROR", "Z_ERROR"}:
            ref = np.array([1e-3 / 21, 0.0, 1e-3 * 20 / 21]) if op.name == "PAULI_CHANNEL_1" else np.array([1e-3])
            for t in op.targets_copy():
                np.testing.assert_allclose(op.gate_args_copy(), ref * scale[t.value])
    unit = rescale_noise_per_qubit(c, np.ones(c.num_qubits))
    dem = lambda x: str(x.detector_error_model(approximate_disjoint_errors=True))
    assert dem(unit) == dem(c)
