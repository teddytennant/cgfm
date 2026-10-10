"""Flow-matching loss and a tiny training step."""

import jax
import jax.numpy as jnp

from cgfm.model import count_params, init_params, paper_config, tiny_config
from cgfm.train import (
    flow_matching_loss,
    init_opt_state,
    make_synthetic_batch,
    train_step,
)


def test_flow_matching_loss_finite():
    key = jax.random.PRNGKey(0)
    key_p, key_b, key_l = jax.random.split(key, 3)
    params = init_params(key_p, tiny_config())
    batch = make_synthetic_batch(key_b, batch_size=4, length=64)
    loss = flow_matching_loss(
        params, batch, key_l, noise_sigma=0.0, dropout_rate=0.0
    )
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))


def test_train_step_changes_parameters():
    key = jax.random.PRNGKey(1)
    key_p, key_b, key_s = jax.random.split(key, 3)
    params = init_params(key_p, tiny_config())
    batch = make_synthetic_batch(key_b, batch_size=4, length=64)
    opt_state = init_opt_state(params)
    new_params, new_state, loss = train_step(
        params,
        batch,
        key_s,
        opt_state,
        noise_sigma=0.0,
        dropout_rate=0.0,
    )
    assert bool(jnp.isfinite(loss))
    leaves0 = jax.tree_util.tree_leaves(params)
    leaves1 = jax.tree_util.tree_leaves(new_params)
    changed = any(not jnp.array_equal(a, b) for a, b in zip(leaves0, leaves1))
    assert changed
    assert new_state is not None


def test_tiny_training_reduces_loss():
    key = jax.random.PRNGKey(0)
    key_p, key_b, key_eval = jax.random.split(key, 3)
    config = tiny_config()
    params = init_params(key_p, config)
    batch = make_synthetic_batch(key_b, batch_size=4, length=64)
    opt_state = init_opt_state(params)
    eval_keys = jax.random.split(key_eval, 4)

    def mean_loss(p):
        losses = [
            flow_matching_loss(
                p, batch, k, noise_sigma=0.0, dropout_rate=0.0
            )
            for k in eval_keys
        ]
        return jnp.mean(jnp.stack(losses))

    loss0 = mean_loss(params)
    for step in range(30):
        step_key = jax.random.fold_in(key, step)
        params, opt_state, _ = train_step(
            params,
            batch,
            step_key,
            opt_state,
            noise_sigma=0.0,
            dropout_rate=config.dropout,
        )
    loss1 = mean_loss(params)
    assert bool(jnp.isfinite(loss0))
    assert bool(jnp.isfinite(loss1))
    assert float(loss1) < float(loss0), (float(loss0), float(loss1))


def test_paper_config_parameter_count():
    params = init_params(jax.random.PRNGKey(0), paper_config())
    n = count_params(params)
    assert 1e5 <= n <= 2e6, n
