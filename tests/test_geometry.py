"""Closed-form cylinder geometry and affine-bridge distortion."""

import jax.numpy as jnp

from cgfm.geometry import (
    affine_amplitude_distortion,
    affine_midpoint_amplitude,
    cylindrical_amplitude_distortion,
    embed,
    geodesic_state,
    oracle_speed_sq,
    oracle_velocity,
    project_cylinder,
    pushforward,
    wrap,
)


def test_wrap_interval_and_branch_point():
    assert wrap(jnp.pi) == jnp.pi
    assert wrap(-jnp.pi) == jnp.pi
    assert wrap(0.0) == 0.0
    assert wrap(3 * jnp.pi) == jnp.pi
    # 2*pi is the identity in exact arithmetic. Float32 is a few ulps off.
    assert jnp.allclose(wrap(2 * jnp.pi), 0.0, atol=1e-5)

    grid = wrap(jnp.linspace(-12.0, 12.0, 8001))
    assert jnp.all(grid > -jnp.pi)
    assert jnp.all(grid <= jnp.pi + 1e-5)
    # A generic angle stays inside the open-closed interval and is 2*pi periodic.
    theta = jnp.array(1.25)
    assert jnp.allclose(wrap(theta), wrap(theta + 4 * jnp.pi), atol=1e-5)
    assert -jnp.pi < wrap(theta) <= jnp.pi


def test_geodesic_endpoints():
    phi0 = jnp.array(0.4)
    a0 = jnp.array(0.8)
    phi1 = jnp.array(2.7)
    a1 = jnp.array(1.5)
    _, _, g0, _, _, _ = geodesic_state(phi0, a0, phi1, a1, 0.0)
    _, _, g1, _, delta_phi, _ = geodesic_state(phi0, a0, phi1, a1, 1.0)
    assert jnp.allclose(g0, embed(phi0, a0), atol=1e-5)
    assert jnp.allclose(g1[..., :2], embed(phi1, a1)[..., :2], atol=1e-5)
    assert jnp.allclose(g1[..., 2], a1, atol=1e-6)
    # Phase representative matches the wrapped geodesic, not a raw jump.
    assert jnp.allclose(g1[..., :2], embed(phi0 + delta_phi, a1)[..., :2], atol=1e-5)

    # Endpoint identity still holds when the raw phase gap exceeds pi.
    phi1_far = phi0 + 4.0
    _, _, g1_far, _, _, _ = geodesic_state(phi0, a0, phi1_far, a1, 1.0)
    assert jnp.allclose(g1_far[..., :2], embed(phi1_far, a1)[..., :2], atol=1e-5)
    assert jnp.allclose(g1_far[..., 2], a1, atol=1e-6)


def test_oracle_norm_constant():
    phi0 = jnp.array(-0.6)
    a0 = jnp.array(1.2)
    phi1 = jnp.array(1.4)
    a1 = jnp.array(0.5)
    expected = oracle_speed_sq(phi0, a0, phi1, a1)
    delta_phi = wrap(phi1 - phi0)
    delta_a = a1 - a0
    assert jnp.allclose(expected, delta_phi ** 2 + delta_a ** 2, atol=1e-6)
    norms = []
    for t in (0.0, 0.2, 0.5, 0.8, 1.0):
        v = oracle_velocity(phi0, a0, phi1, a1, t)
        norms.append(jnp.sum(v ** 2))
    norms = jnp.stack(norms)
    assert jnp.allclose(norms, expected, atol=1e-5)
    assert jnp.max(jnp.abs(norms - norms[0])) < 1e-5


def test_pushforward_is_tangent():
    phi = jnp.array(0.85)
    amplitude = jnp.array(1.4)
    g = embed(phi, amplitude)
    omega = jnp.array(-0.7)
    a_dot = jnp.array(0.25)
    v = pushforward(g, omega, a_dot)
    tangent = v[..., 0] * jnp.cos(phi) + v[..., 1] * jnp.sin(phi)
    assert jnp.allclose(tangent, 0.0, atol=1e-5)
    expected = jnp.stack(
        [-jnp.sin(phi) * omega, jnp.cos(phi) * omega, a_dot],
        axis=-1,
    )
    assert jnp.allclose(v, expected, atol=1e-5)

    # Batched states stay tangent too.
    phis = jnp.array([0.2, -1.1, 2.4])
    amps = jnp.array([0.5, 1.0, 2.0])
    g_b = embed(phis, amps)
    v_b = pushforward(g_b, jnp.array([0.1, -0.3, 0.8]), jnp.array([0.0, 0.2, -0.4]))
    tangent_b = jnp.sum(v_b[..., :2] * g_b[..., :2], axis=-1)
    assert jnp.allclose(tangent_b, 0.0, atol=1e-5)


def test_project_cylinder():
    g = jnp.array([3.0, 4.0, 2.0])
    projected = project_cylinder(g)
    assert jnp.allclose(jnp.linalg.norm(projected[:2]), 1.0, atol=1e-5)
    assert jnp.allclose(projected, jnp.array([0.6, 0.8, 2.0]), atol=1e-5)

    tiny_amp = project_cylinder(jnp.array([0.0, -5.0, 1e-20]))
    assert jnp.allclose(jnp.linalg.norm(tiny_amp[:2]), 1.0, atol=1e-5)
    assert tiny_amp[2] >= 1e-12
    assert jnp.allclose(tiny_amp[2], 1e-12, atol=1e-6)

    negative = project_cylinder(jnp.array([1.0, 0.0, -0.4]))
    assert negative[2] >= 1e-12
    assert jnp.allclose(jnp.linalg.norm(negative[:2]), 1.0, atol=1e-5)

    zero_circ = project_cylinder(jnp.array([0.0, 0.0, 0.7]), eps=1e-12)
    assert jnp.allclose(jnp.linalg.norm(zero_circ[:2]), 1.0, atol=1e-5)
    assert jnp.allclose(zero_circ[2], 0.7, atol=1e-6)


def test_cylindrical_amplitude_distortion_is_zero():
    d_amp = cylindrical_amplitude_distortion(0.2, 1.0, 1.5, 1.0)
    assert abs(float(d_amp)) < 1e-6
    d_amp_other = cylindrical_amplitude_distortion(-0.4, 0.7, 2.2, 0.7, a_ref=0.7)
    assert abs(float(d_amp_other)) < 1e-6


def test_affine_amplitude_distortion_positive():
    d_amp = affine_amplitude_distortion(0.0, 1.0, jnp.pi / 2, 1.0)
    assert float(d_amp) > 1e-3


def test_affine_leading_order():
    delta_phi = 0.2
    d_amp = affine_amplitude_distortion(0.0, 1.0, delta_phi, 1.0)
    expected = (delta_phi ** 4) / 120.0
    rel = abs(float(d_amp) - expected) / expected
    assert rel < 0.05, (float(d_amp), expected, rel)


def test_affine_midpoint_amplitude_vanishes_at_pi():
    mid = affine_midpoint_amplitude(0.3, 1.0, 0.3 + jnp.pi, 1.0)
    assert abs(float(mid)) < 1e-5
    mid_neg = affine_midpoint_amplitude(-0.8, 1.4, -0.8 - jnp.pi, 1.4)
    assert abs(float(mid_neg)) < 1e-5
    # Equal amplitudes and a half-turn chord pass through the origin.
    mid_zero_phase = affine_midpoint_amplitude(0.0, 1.0, jnp.pi, 1.0)
    assert abs(float(mid_zero_phase)) < 1e-5
