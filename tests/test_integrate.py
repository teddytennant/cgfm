"""Heun integration of the oracle field with cylinder projection."""

import jax.numpy as jnp

from cgfm.geometry import embed, oracle_field, wrap
from cgfm.integrate import integrate_heun, waveform_from_phase


def _endpoint_error(phi0, a0, phi1, a1, n_steps=32):
    g0 = embed(phi0, a0)
    delta_phi = wrap(phi1 - phi0)
    g1_true = embed(phi0 + delta_phi, a1)
    v_fn = oracle_field(phi0, a0, phi1, a1)
    g1, trajectory = integrate_heun(
        v_fn, g0, n_steps, return_trajectory=True
    )
    return g1, g1_true, trajectory


def test_heun_oracle_reaches_target():
    cases = [
        (0.2, 1.0, 1.1, 1.4),
        (0.0, 1.0, float(jnp.pi), 1.0),
        (-0.5, 0.6, 2.2, 1.8),
        (1.0, 1.3, 1.0 - 2.4, 0.5),
    ]
    for phi0, a0, phi1, a1 in cases:
        g1, g1_true, trajectory = _endpoint_error(
            jnp.asarray(phi0),
            jnp.asarray(a0),
            jnp.asarray(phi1),
            jnp.asarray(a1),
            n_steps=32,
        )
        assert jnp.allclose(g1[..., :2], g1_true[..., :2], atol=1e-2)
        assert jnp.allclose(g1[..., 2], g1_true[..., 2], atol=1e-2)
        circ = jnp.linalg.norm(trajectory[..., :2], axis=-1)
        assert jnp.allclose(circ, 1.0, atol=1e-4)
        assert jnp.all(trajectory[..., 2] > 0.0)


def test_heun_waveform_batch_stays_on_cylinder():
    length = 16
    n = jnp.arange(length)
    phi0 = 0.15 * n
    a0 = 0.8 + 0.02 * n
    phi1 = phi0 + 0.9
    a1 = a0 + 0.3
    g1, g1_true, trajectory = _endpoint_error(phi0, a0, phi1, a1, n_steps=32)
    assert g1.shape == (length, 3)
    assert jnp.allclose(g1[..., :2], g1_true[..., :2], atol=1e-2)
    assert jnp.allclose(g1[..., 2], g1_true[..., 2], atol=1e-2)
    circ = jnp.linalg.norm(trajectory[..., :2], axis=-1)
    assert jnp.allclose(circ, 1.0, atol=1e-4)
    assert jnp.all(trajectory[..., 2] > 0.0)

    rebuilt = waveform_from_phase(a1, phi1)
    assert rebuilt.shape == (length,)
    assert jnp.allclose(rebuilt, a1 * jnp.cos(phi1), atol=1e-6)
