"""Cylinder geometry for the phase-amplitude geodesic bridge.

States live on S^1 x R>0, embedded as g = (cos phi, sin phi, a) with a > 0.
The supervised path is the closed-form geodesic, not the Euclidean chord.
"""

import jax.numpy as jnp


def wrap(theta):
    """Unique representative of theta mod 2*pi in (-pi, pi].

    The usual ((theta + pi) mod 2*pi) - pi lands in [-pi, pi). The single
    point -pi is mapped to +pi so the interval matches the cylinder chart.
    wrap(pi) == pi, wrap(-pi) == pi, wrap(0) == 0, wrap(3*pi) == pi.
    """
    theta = jnp.asarray(theta)
    pi = jnp.array(jnp.pi, dtype=theta.dtype)
    y = jnp.mod(theta + pi, 2 * pi) - pi
    return jnp.where(y == -pi, pi, y)


def circular_distance(phi1, phi2):
    """Absolute circular distance in [0, pi].

    circ(phi1, phi2) = abs( ((phi1 - phi2 + pi) mod 2*pi) - pi ).
    """
    diff = jnp.asarray(phi1) - jnp.asarray(phi2)
    pi = jnp.array(jnp.pi, dtype=diff.dtype)
    return jnp.abs(jnp.mod(diff + pi, 2 * pi) - pi)


def ht_distance(a_pred, phi_pred, a_tgt, phi_tgt):
    """Mean Hilbert-cylinder distance over samples.

    mean sqrt( (a_pred - a_tgt)^2 + circ(phi_pred, phi_tgt)^2 ).
    """
    err = jnp.sqrt(
        (jnp.asarray(a_pred) - jnp.asarray(a_tgt)) ** 2
        + circular_distance(phi_pred, phi_tgt) ** 2
    )
    return jnp.mean(err)


def embed(phi, amplitude):
    """Embed phase and amplitude as (cos phi, sin phi, a)."""
    phi = jnp.asarray(phi)
    amplitude = jnp.asarray(amplitude)
    return jnp.stack([jnp.cos(phi), jnp.sin(phi), amplitude], axis=-1)


def phase_from_embedding(g):
    """phi = atan2(g[..., 1], g[..., 0])."""
    g = jnp.asarray(g)
    return jnp.arctan2(g[..., 1], g[..., 0])


def deltas(phi0, a0, phi1, a1):
    """Wrapped phase gap and raw amplitude gap."""
    delta_phi = wrap(jnp.asarray(phi1) - jnp.asarray(phi0))
    delta_a = jnp.asarray(a1) - jnp.asarray(a0)
    return delta_phi, delta_a


def intrinsic_length(phi0, a0, phi1, a1):
    """L_C = sqrt(delta_phi^2 + delta_a^2)."""
    delta_phi, delta_a = deltas(phi0, a0, phi1, a1)
    return jnp.sqrt(delta_phi ** 2 + delta_a ** 2)


def geodesic_state(phi0, a0, phi1, a1, t):
    """Closed-form geodesic state and oracle velocity at time t.

    phi_t = phi0 + t * delta_phi
    a_t = a0 + t * delta_a
    g_t = (cos phi_t, sin phi_t, a_t)
    v*_t = (-sin(phi_t) * delta_phi, cos(phi_t) * delta_phi, delta_a)

    t broadcasts with the phase arrays. Returns phi_t, a_t, g_t, v_star,
    delta_phi, delta_a.
    """
    delta_phi, delta_a = deltas(phi0, a0, phi1, a1)
    t = jnp.asarray(t)
    phi_t = jnp.asarray(phi0) + t * delta_phi
    a_t = jnp.asarray(a0) + t * delta_a
    g_t = embed(phi_t, a_t)
    delta_a_b = jnp.broadcast_to(delta_a, jnp.shape(phi_t))
    v_star = jnp.stack(
        [
            -jnp.sin(phi_t) * delta_phi,
            jnp.cos(phi_t) * delta_phi,
            delta_a_b,
        ],
        axis=-1,
    )
    return phi_t, a_t, g_t, v_star, delta_phi, delta_a


def oracle_velocity(phi0, a0, phi1, a1, t):
    """Ambient oracle velocity v*_t along the geodesic."""
    return geodesic_state(phi0, a0, phi1, a1, t)[3]


def oracle_speed_sq(phi0, a0, phi1, a1):
    """Constant squared Euclidean norm of v*_t: delta_phi^2 + delta_a^2."""
    delta_phi, delta_a = deltas(phi0, a0, phi1, a1)
    return delta_phi ** 2 + delta_a ** 2


def oracle_field(phi0, a0, phi1, a1):
    """v_fn(g, t, cond) for the closed-form oracle field.

    The field depends on t and the endpoints, not on the current embedding.
    """
    delta_phi, delta_a = deltas(phi0, a0, phi1, a1)

    def v_fn(g, t, cond=None):
        del g, cond
        phi_t = jnp.asarray(phi0) + t * delta_phi
        delta_a_b = jnp.broadcast_to(delta_a, jnp.shape(phi_t))
        return jnp.stack(
            [
                -jnp.sin(phi_t) * delta_phi,
                jnp.cos(phi_t) * delta_phi,
                delta_a_b,
            ],
            axis=-1,
        )

    return v_fn


def pushforward(g, omega, a_dot):
    """Map intrinsic velocity (omega, a_dot) to an ambient tangent vector.

    phi = atan2(g[..., 1], g[..., 0])
    v = (-sin(phi) * omega, cos(phi) * omega, a_dot)

    The circular part of v is orthogonal to (cos phi, sin phi).
    """
    phi = phase_from_embedding(g)
    return jnp.stack(
        [
            -jnp.sin(phi) * omega,
            jnp.cos(phi) * omega,
            a_dot,
        ],
        axis=-1,
    )


def project_cylinder(g, eps=1e-12):
    """Project an ambient state back onto the cylinder.

    Renormalize the circular component (g0, g1) to unit L2 norm. If that
    component is numerically zero, fall back to (1, 0).

    Amplitude g2 is left unchanged when it is already >= eps. Values below
    eps are clamped to eps. That clamp is a numerical guard so the state
    stays in R>0. The paper only states the circular renormalization.
    """
    g = jnp.asarray(g)
    circ = g[..., :2]
    norm = jnp.sqrt(jnp.sum(circ * circ, axis=-1, keepdims=True))
    default = jnp.zeros_like(circ).at[..., 0].set(jnp.array(1.0, dtype=g.dtype))
    circ_unit = jnp.where(norm > eps, circ / jnp.maximum(norm, eps), default)
    amplitude = jnp.maximum(g[..., 2:3], jnp.array(eps, dtype=g.dtype))
    return jnp.concatenate([circ_unit, amplitude], axis=-1)


def affine_path(phi0, a0, phi1, a1, t):
    """Euclidean chord in the complex plane, then polar coordinates.

    w0 = a0 * exp(i phi0)
    w1 = a1 * exp(i (phi0 + delta_phi))
    with the same wrapped delta_phi used by the geodesic.
    gamma_aff(t) = (1 - t) * w0 + t * w1
    a_aff = abs(gamma_aff), phi_aff = angle(gamma_aff)

    Returns phi_aff, a_aff, embedded state (cos, sin, a_aff), and the
    ambient time derivative of that embedding. The derivative is guarded
    when the chord passes through the origin (amplitude near 0).
    """
    delta_phi, _ = deltas(phi0, a0, phi1, a1)
    phi0 = jnp.asarray(phi0)
    a0 = jnp.asarray(a0)
    a1 = jnp.asarray(a1)
    t = jnp.asarray(t)
    w0 = a0 * jnp.exp(1j * phi0)
    w1 = a1 * jnp.exp(1j * (phi0 + delta_phi))
    gamma = (1 - t) * w0 + t * w1
    dgamma = w1 - w0
    amplitude = jnp.abs(gamma)
    phi = jnp.angle(gamma)
    denom = jnp.maximum(amplitude, jnp.array(1e-8, dtype=amplitude.dtype))
    unit = gamma / denom
    a_dot = jnp.real(jnp.conj(unit) * dgamma)
    phi_dot = jnp.imag(jnp.conj(unit) * dgamma) / denom
    # At a vanishing midpoint the angle is undefined. Zero the angular rate.
    small = amplitude < jnp.array(1e-8, dtype=amplitude.dtype)
    phi_dot = jnp.where(small, jnp.zeros_like(phi_dot), phi_dot)
    a_dot = jnp.where(small, jnp.zeros_like(a_dot), a_dot)
    g = embed(phi, amplitude)
    velocity = jnp.stack(
        [
            -jnp.sin(phi) * phi_dot,
            jnp.cos(phi) * phi_dot,
            a_dot,
        ],
        axis=-1,
    )
    return phi, amplitude, g, velocity


def affine_midpoint_amplitude(phi0, a0, phi1, a1):
    """Amplitude of the affine chord at t = 1/2."""
    _, amplitude, _, _ = affine_path(phi0, a0, phi1, a1, 0.5)
    return amplitude


def _trapezoid(y, t):
    """Trapezoidal integral of y along the last axis of t."""
    y = jnp.asarray(y)
    t = jnp.asarray(t)
    dt = jnp.diff(t)
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dt, axis=-1)


def _sample_t(n):
    return jnp.linspace(0.0, 1.0, n)


def cylindrical_amplitude_distortion(phi0, a0, phi1, a1, a_ref=None, n=4097):
    """D_amp = integral_0^1 (a_t - a_ref)^2 dt along the geodesic.

    When a0 == a1 == a_ref this integral is 0.
    """
    t = _sample_t(n)
    _, a_t, _, _, _, _ = geodesic_state(phi0, a0, phi1, a1, t)
    if a_ref is None:
        a_ref = a0
    return _trapezoid((a_t - a_ref) ** 2, t)


def cylindrical_frequency_distortion(phi0, a0, phi1, a1, n=4097):
    """D_freq = integral_0^1 (dphi/dt - delta_phi)^2 dt. Zero on the geodesic."""
    t = _sample_t(n)
    delta_phi, _ = deltas(phi0, a0, phi1, a1)
    phi_dot = jnp.broadcast_to(delta_phi, t.shape)
    return _trapezoid((phi_dot - delta_phi) ** 2, t)


def affine_amplitude_distortion(phi0, a0, phi1, a1, a_ref=None, n=4097):
    """D_amp of the affine chord, trapezoid quadrature on n points.

    Leading order as delta_phi -> 0 with a0 == a1 == a:
    D_amp = a^2 * delta_phi^4 / 120 + O(delta_phi^6).
    """
    t = _sample_t(n)
    _, amplitude, _, _ = affine_path(phi0, a0, phi1, a1, t)
    if a_ref is None:
        a_ref = a0
    return _trapezoid((amplitude - a_ref) ** 2, t)


def affine_frequency_distortion(phi0, a0, phi1, a1, n=4097):
    """D_freq of the affine chord against the wrapped phase gap."""
    t = _sample_t(n)
    delta_phi, _ = deltas(phi0, a0, phi1, a1)
    phi_dot = _affine_phi_dot(phi0, a0, phi1, a1, t)
    return _trapezoid((phi_dot - delta_phi) ** 2, t)


def _affine_phi_dot(phi0, a0, phi1, a1, t):
    delta_phi, _ = deltas(phi0, a0, phi1, a1)
    phi0 = jnp.asarray(phi0)
    a0 = jnp.asarray(a0)
    a1 = jnp.asarray(a1)
    w0 = a0 * jnp.exp(1j * phi0)
    w1 = a1 * jnp.exp(1j * (phi0 + delta_phi))
    gamma = (1 - t) * w0 + t * w1
    dgamma = w1 - w0
    amplitude = jnp.abs(gamma)
    denom = jnp.maximum(amplitude, jnp.array(1e-8, dtype=amplitude.dtype))
    phi_dot = jnp.imag(jnp.conj(gamma / denom) * dgamma) / denom
    return jnp.where(amplitude < 1e-8, jnp.zeros_like(phi_dot), phi_dot)


def reconstruct(amplitude, phase):
    """Closed-form waveform a * cos(phi)."""
    return jnp.asarray(amplitude) * jnp.cos(jnp.asarray(phase))
