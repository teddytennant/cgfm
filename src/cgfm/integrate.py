"""Heun integration on the cylinder embedding.

After every substep (predictor and corrector) the state is projected
back onto the cylinder.
"""

import jax
import jax.numpy as jnp

from cgfm.geometry import project_cylinder, reconstruct


def integrate_heun(v_fn, g0, n_steps, cond=None, return_trajectory=False):
    """Improved Euler (Heun) from t=0 to t=1.

    v_fn(g, t, cond) returns an ambient velocity with the same shape as g.
    Predictor and corrector states are passed through project_cylinder.
    n_steps is a Python int.

    Returns g at t=1. If return_trajectory is true, also returns the
    initial state and every projected substep, stacked on a leading axis.
    """
    g0 = jnp.asarray(g0)
    dt = jnp.asarray(1.0 / n_steps, dtype=g0.dtype)

    def step(g, i):
        t = i.astype(g.dtype) * dt
        v1 = v_fn(g, t, cond)
        g_pred = project_cylinder(g + dt * v1)
        v2 = v_fn(g_pred, t + dt, cond)
        g_next = project_cylinder(g + 0.5 * dt * (v1 + v2))
        return g_next, jnp.stack([g_pred, g_next], axis=0)

    g1, packed = jax.lax.scan(step, g0, jnp.arange(n_steps))
    if not return_trajectory:
        return g1
    flat = packed.reshape((n_steps * 2,) + packed.shape[2:])
    trajectory = jnp.concatenate([g0[None], flat], axis=0)
    return g1, trajectory


def waveform_from_phase(amplitude, phase):
    """Closed-form reconstruction: waveform = a * cos(phi)."""
    return reconstruct(amplitude, phase)


def waveform_from_embedding(g):
    """Reconstruct a * cos(phi) from a cylinder embedding."""
    phi = jnp.arctan2(g[..., 1], g[..., 0])
    return reconstruct(g[..., 2], phi)
