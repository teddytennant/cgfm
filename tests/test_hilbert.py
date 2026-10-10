"""FFT Hilbert analytic signal."""

import jax
import jax.numpy as jnp

from cgfm.hilbert import amplitude_phase, analytic_signal


def test_real_part_matches_input():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 96))
    z = analytic_signal(x)
    assert jnp.allclose(jnp.real(z), x, atol=1e-5)

    t = jnp.arange(64)
    cosine = jnp.cos(2 * jnp.pi * 3 * t / 64)
    z_cos = analytic_signal(cosine)
    assert jnp.allclose(jnp.real(z_cos), cosine, atol=1e-5)


def test_cosine_integer_bin_amplitude_and_phase():
    length = 128
    k = 5
    n = jnp.arange(length)
    x = jnp.cos(2 * jnp.pi * k * n / length)
    amplitude, phase = amplitude_phase(x)
    assert jnp.allclose(amplitude, 1.0, atol=1e-4)
    expected = 2 * jnp.pi * k * n / length
    # Analytic signal of cos is exp(i w t), so the global offset is ~0.
    assert jnp.allclose(phase, expected, atol=1e-3)
    advance = phase[-1] - phase[0]
    # One sample short of a full k cycles, which is about 2*pi*k.
    assert abs(float(advance - 2 * jnp.pi * k * (length - 1) / length)) < 1e-3
    assert abs(float(advance) - 2 * jnp.pi * k) / (2 * jnp.pi * k) < 0.02
