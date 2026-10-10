"""Conditional flow matching on the cylinder geodesic.

The loss supervises the ambient pushforward of the network field against
the closed-form oracle velocity. Training may perturb the raw source
waveform with N(0, sigma^2) before the Hilbert embedding so the geometry
stays on the cylinder. That noise is training-only.

train_step uses AdamW (beta1=0.9, beta2=0.99, weight_decay=1e-2) with
global gradient clipping at 1.0. A cosine schedule helper is provided.
The step itself uses a constant learning rate so short tests are stable.
The paper schedule runs for many thousands of epochs and is not required
here.
"""

import jax
import jax.numpy as jnp
import optax

from cgfm.geometry import embed, geodesic_state, pushforward
from cgfm.hilbert import amplitude_phase
from cgfm.model import apply

LEARNING_RATE = 3e-3

_OPTIMIZER = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        learning_rate=LEARNING_RATE,
        b1=0.9,
        b2=0.99,
        weight_decay=1e-2,
    ),
)


def cosine_lr_schedule(step, total_steps=20_000, lr_max=1e-3, lr_min=1e-6):
    """Cosine decay from lr_max to lr_min. Not used inside train_step."""
    step = jnp.asarray(step, dtype=jnp.float32)
    progress = jnp.clip(step / float(total_steps), 0.0, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * progress))


def init_opt_state(params):
    """AdamW state matching train_step."""
    return _OPTIMIZER.init(params)


def make_synthetic_batch(key, batch_size=4, length=64):
    """Quasiperiodic source/target pairs plus a conditioning sinusoid.

    x(t) = (1 + b sin(2*pi*t)) * sin(2*pi*f*t + phase)
    Target uses a phase shift and an amplitude scale. The conditioning
    sinusoid carries that same phase shift and scale so the map is
    determined by the inputs.
    """
    k_f, k_phase, k_dphi, k_amp, k_b = jax.random.split(key, 5)
    t = jnp.arange(length, dtype=jnp.float32) / jnp.float32(length)
    freq = jax.random.randint(k_f, (batch_size,), 2, 7).astype(jnp.float32)
    phase = jax.random.uniform(k_phase, (batch_size,), minval=0.0, maxval=2.0 * jnp.pi)
    dphase = jax.random.uniform(k_dphi, (batch_size,), minval=-1.2, maxval=1.2)
    amp_scale = jax.random.uniform(k_amp, (batch_size,), minval=0.6, maxval=1.5)
    envelope = jax.random.uniform(k_b, (batch_size,), minval=0.1, maxval=0.35)

    tt = t[None, :]
    env = 1.0 + envelope[:, None] * jnp.sin(2.0 * jnp.pi * tt)
    x0 = env * jnp.sin(2.0 * jnp.pi * freq[:, None] * tt + phase[:, None])
    x1 = (
        amp_scale[:, None]
        * env
        * jnp.sin(2.0 * jnp.pi * freq[:, None] * tt + phase[:, None] + dphase[:, None])
    )
    cond = amp_scale[:, None] * jnp.sin(
        2.0 * jnp.pi * tt + phase[:, None] + dphase[:, None]
    )
    return {"x0": x0, "x1": x1, "cond": cond}


def flow_matching_loss(params, batch, key, noise_sigma=0.0, dropout_rate=0.0):
    """Mean squared error between pushed-forward field and oracle velocity.

    t is drawn uniformly from [0, 1), one draw per batch element.
    The 9-channel input is embed(g_t), embed(source), embed(cond).
    noise_sigma and dropout_rate are scalars. Tests pass 0 to disable
    training noise and dropout. Zero noise leaves the source waveform
    unchanged, then the Hilbert embedding is rebuilt.
    """
    key_t, key_noise, key_drop = jax.random.split(key, 3)
    x0 = batch["x0"]
    x1 = batch["x1"]
    cond = batch["cond"]
    # Always add the noise term so sigma can be a traced scalar. Zero sigma
    # leaves the waveform unchanged.
    sigma = jnp.asarray(noise_sigma, dtype=x0.dtype)
    x0 = x0 + sigma * jax.random.normal(key_noise, x0.shape)

    a0, phi0 = amplitude_phase(x0)
    a1, phi1 = amplitude_phase(x1)
    a_c, phi_c = amplitude_phase(cond)

    t = jax.random.uniform(key_t, (x0.shape[0],), dtype=x0.dtype)
    _, _, g_t, v_star, _, _ = geodesic_state(phi0, a0, phi1, a1, t[:, None])
    g_t = jax.lax.stop_gradient(g_t)
    v_star = jax.lax.stop_gradient(v_star)
    g_source = jax.lax.stop_gradient(embed(phi0, a0))
    g_cond = jax.lax.stop_gradient(embed(phi_c, a_c))
    inputs = jnp.concatenate([g_t, g_source, g_cond], axis=-1)
    omega, a_dot = apply(
        params, inputs, t, key=key_drop, dropout_rate=dropout_rate
    )
    v_hat = pushforward(g_t, omega, a_dot)
    return jnp.mean((v_hat - v_star) ** 2)


def train_step(params, batch, key, opt_state, noise_sigma=0.05, dropout_rate=0.05):
    """One AdamW update. Jit-compiled.

    Returns updated params, updated opt_state, and the scalar loss.
    Default noise_sigma is 0.05 (training only). Pass 0 to disable it.
    """
    params, opt_state, loss = _train_step(
        params, batch, key, opt_state, noise_sigma, dropout_rate
    )
    return params, opt_state, loss


@jax.jit
def _train_step(params, batch, key, opt_state, noise_sigma, dropout_rate):
    def loss_fn(p):
        return flow_matching_loss(
            p,
            batch,
            key,
            noise_sigma=noise_sigma,
            dropout_rate=dropout_rate,
        )

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = _OPTIMIZER.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
