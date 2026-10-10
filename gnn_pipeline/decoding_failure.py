"""The decoding-failure criterion used by every evaluation path.

A shot is a failure if the decoder's estimate does not reproduce the measured
syndrome, or if it does and the residual error flips at least one logical
observable. The syndrome test is not optional: when the estimate misses the
syndrome, the residual is not a logical operator, and its parity against a
fixed set of logical representatives depends on which representatives were
chosen rather than on anything the decoder did.

For CSS codes the two components are tested separately and combined with OR,
so a shot fails if either component fails. Observable arrays follow the layout
written by ``generate_codecap``: the first ``lx.shape[0]`` columns are X-type
logical parities (flipped by Z errors), the remaining ``lz.shape[0]`` columns
are Z-type logical parities (flipped by X errors).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FailureReport:
    """Per-shot failure flags. The two categories are disjoint."""

    syndrome_mismatch: np.ndarray
    logical_flip: np.ndarray

    @property
    def failure(self) -> np.ndarray:
        return self.syndrome_mismatch | self.logical_flip

    def __or__(self, other: "FailureReport") -> "FailureReport":
        mismatch = self.syndrome_mismatch | other.syndrome_mismatch
        return FailureReport(
            syndrome_mismatch=mismatch,
            logical_flip=(self.logical_flip | other.logical_flip) & ~mismatch,
        )

    def counts(self) -> dict:
        return {
            "failures": int(self.failure.sum()),
            "syndrome_mismatch": int(self.syndrome_mismatch.sum()),
            "logical_flip": int(self.logical_flip.sum()),
            "shots": int(self.failure.shape[0]),
        }


def _bits(a, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == bool:
        return a.astype(np.int64)
    rounded = np.rint(a)
    if not np.all((rounded == 0) | (rounded == 1)) or not np.allclose(a, rounded, atol=1e-6):
        raise ValueError(f"{name} must contain only 0/1 values (got hard decisions? marginals?)")
    return rounded.astype(np.int64)


def _as_batch(a: np.ndarray) -> np.ndarray:
    return a[np.newaxis, :] if a.ndim == 1 else a


def component_failures(
    estimate,
    syndrome,
    check_matrix,
    logicals,
    observables,
) -> FailureReport:
    """Failures for one decoding problem ``check_matrix @ e = syndrome``.

    Args:
        estimate: (B, n) or (n,) decoder hard decisions.
        syndrome: (B, m) or (m,) measured syndrome.
        check_matrix: (m, n) parity-check matrix the decoder solved.
        logicals: (k, n) logical operators whose parity defines the observables.
        observables: (B, k) or (k,) true logical parities of the physical error.
    """
    e = _as_batch(_bits(estimate, "estimate"))
    s = _as_batch(_bits(syndrome, "syndrome"))
    H = _bits(check_matrix, "check_matrix")
    L = _bits(logicals, "logicals")
    obs = _as_batch(_bits(observables, "observables"))

    if H.shape[1] != e.shape[1] or L.shape[1] != e.shape[1]:
        raise ValueError(f"estimate has {e.shape[1]} columns; check_matrix {H.shape}, logicals {L.shape}")
    if s.shape != (e.shape[0], H.shape[0]):
        raise ValueError(f"syndrome shape {s.shape} does not match ({e.shape[0]}, {H.shape[0]})")
    if obs.shape != (e.shape[0], L.shape[0]):
        raise ValueError(f"observables shape {obs.shape} does not match ({e.shape[0]}, {L.shape[0]})")

    mismatch = ((e @ H.T) & 1 != s).any(axis=1)
    flip = ((e @ L.T) & 1 != obs).any(axis=1)
    return FailureReport(syndrome_mismatch=mismatch, logical_flip=flip & ~mismatch)


def css_failures(
    z_estimate,
    x_estimate,
    x_syndrome,
    z_syndrome,
    hx,
    hz,
    lx,
    lz,
    observables,
) -> FailureReport:
    """Failures for separate CSS decoding, with components combined by OR.

    ``hx @ z_estimate`` must reproduce ``x_syndrome`` and ``hz @ x_estimate``
    must reproduce ``z_syndrome``; ``lx`` measures Z errors and ``lz`` measures
    X errors.
    """
    obs = _as_batch(np.asarray(observables))
    kx, kz = np.asarray(lx).shape[0], np.asarray(lz).shape[0]
    if obs.shape[1] != kx + kz:
        raise ValueError(f"observables have {obs.shape[1]} columns, expected kx + kz = {kx} + {kz}")
    z_part = component_failures(z_estimate, x_syndrome, hx, lx, obs[:, :kx])
    x_part = component_failures(x_estimate, z_syndrome, hz, lz, obs[:, kx:])
    return z_part | x_part


def css_failures_from_errors(
    z_estimate,
    x_estimate,
    z_true,
    x_true,
    hx,
    hz,
    lx,
    lz,
) -> FailureReport:
    """Same criterion as :func:`css_failures`, with the syndrome and
    observables derived from the true physical errors."""
    z_true = _as_batch(_bits(z_true, "z_true"))
    x_true = _as_batch(_bits(x_true, "x_true"))
    hx, hz, lx, lz = (_bits(m, name) for m, name in ((hx, "hx"), (hz, "hz"), (lx, "lx"), (lz, "lz")))
    observables = np.concatenate([(z_true @ lx.T) & 1, (x_true @ lz.T) & 1], axis=1)
    return css_failures(
        z_estimate, x_estimate,
        (z_true @ hx.T) & 1, (x_true @ hz.T) & 1,
        hx, hz, lx, lz, observables,
    )
