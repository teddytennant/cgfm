"""1D U-Net for the cylindrical conditional vector field.

The network maps 9 channels to 2 intrinsic outputs (omega, a_dot).
Channels are the cylinder embeddings of the current state, the source,
and the conditioning waveform, concatenated in that order.

Time t in [0, 1] is encoded with a DDPM-style sinusoidal embedding,
scaled by 1000, passed through a two-layer MLP, and added (after a
linear projection) inside every residual block, including the bottleneck.

Dropout, when enabled, is applied only to the concatenated input.
The intrinsic pushforward to ambient velocity is applied outside the
network, before the flow-matching loss.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    widths: tuple
    dropout: float = 0.05
    time_dim: int = 128
    time_hidden: int = 512
    kernel_size: int = 3
    res_blocks: int = 2
    in_channels: int = 9
    out_channels: int = 2


def paper_config():
    """Four levels, widths (32, 32, 64, 64), 5 percent input dropout.

    Parameter count is on the order of 1e6 with these widths.
    """
    return Config(
        widths=(32, 32, 64, 64),
        dropout=0.05,
        time_dim=128,
        time_hidden=512,
        kernel_size=3,
        res_blocks=2,
    )


def tiny_config():
    """Small CPU config: widths (8, 8), dropout off."""
    return Config(
        widths=(8, 8),
        dropout=0.0,
        time_dim=32,
        time_hidden=32,
        kernel_size=3,
        res_blocks=1,
    )


class _Keys:
    def __init__(self, key):
        self.key = key

    def next(self):
        self.key, sub = jax.random.split(self.key)
        return sub


def _init_conv(key, kernel, cin, cout, scale=1.0):
    std = scale * (2.0 / (kernel * cin)) ** 0.5
    weight = jax.random.normal(key, (kernel, cin, cout)) * std
    bias = jnp.zeros((cout,))
    return {"w": weight, "b": bias}


def _init_linear(key, din, dout, scale=1.0):
    std = scale * (2.0 / din) ** 0.5
    weight = jax.random.normal(key, (din, dout)) * std
    bias = jnp.zeros((dout,))
    return {"w": weight, "b": bias}


def _init_gn(channels):
    return {
        "scale": jnp.ones((channels,)),
        "bias": jnp.zeros((channels,)),
    }


def _init_res(keys, cin, cout, temb_dim, kernel):
    block = {
        "conv1": _init_conv(keys.next(), kernel, cin, cout),
        "gn1": _init_gn(cout),
        "tproj": _init_linear(keys.next(), temb_dim, cout),
        "conv2": _init_conv(keys.next(), kernel, cout, cout),
        "gn2": _init_gn(cout),
    }
    if cin != cout:
        block["skip"] = _init_conv(keys.next(), 1, cin, cout)
    return block


def init_params(key, config):
    """Initialize a U-Net pytree for `config`."""
    keys = _Keys(key)
    widths = tuple(config.widths)
    time_1 = _init_linear(keys.next(), config.time_dim, config.time_hidden)
    time_2 = _init_linear(keys.next(), config.time_hidden, config.time_hidden)
    params = {
        "time": {
            "w1": time_1["w"],
            "b1": time_1["b"],
            "w2": time_2["w"],
            "b2": time_2["b"],
        }
    }

    channels = config.in_channels
    params["stem"] = _init_conv(keys.next(), config.kernel_size, channels, widths[0])
    channels = widths[0]

    encoder = []
    for i, width in enumerate(widths):
        blocks = []
        for _ in range(config.res_blocks):
            blocks.append(
                _init_res(keys, channels, width, config.time_hidden, config.kernel_size)
            )
            channels = width
        level = {"blocks": blocks}
        if i < len(widths) - 1:
            level["down"] = _init_conv(
                keys.next(), config.kernel_size, channels, widths[i + 1]
            )
            channels = widths[i + 1]
        encoder.append(level)
    params["enc"] = encoder

    mid = []
    for _ in range(config.res_blocks):
        mid.append(
            _init_res(keys, channels, widths[-1], config.time_hidden, config.kernel_size)
        )
        channels = widths[-1]
    params["mid"] = {"blocks": mid}

    decoder = []
    for i in reversed(range(len(widths))):
        cin = channels + widths[i]
        blocks = []
        for _ in range(config.res_blocks):
            blocks.append(
                _init_res(keys, cin, widths[i], config.time_hidden, config.kernel_size)
            )
            cin = widths[i]
        decoder.append({"blocks": blocks})
        channels = widths[i]
    params["dec"] = decoder
    # Small head so the initial intrinsic velocity is near zero.
    params["head"] = _init_conv(
        keys.next(),
        config.kernel_size,
        channels,
        config.out_channels,
        scale=1e-2,
    )
    return params


def count_params(params):
    """Number of scalar parameters in a pytree."""
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def _groups(channels):
    groups = min(8, int(channels))
    while channels % groups != 0:
        groups -= 1
    return groups


def _group_norm(x, scale, bias, eps=1e-5):
    batch, length, channels = x.shape
    groups = _groups(channels)
    grouped = x.reshape(batch, length, groups, channels // groups)
    mean = jnp.mean(grouped, axis=(1, 3), keepdims=True)
    var = jnp.mean((grouped - mean) ** 2, axis=(1, 3), keepdims=True)
    grouped = (grouped - mean) * jax.lax.rsqrt(var + eps)
    y = grouped.reshape(batch, length, channels)
    return y * scale + bias


def _conv1d(x, params, stride=1):
    y = jax.lax.conv_general_dilated(
        x,
        params["w"],
        window_strides=(stride,),
        padding="SAME",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    return y + params["b"]


def _upsample(x):
    return jnp.repeat(x, 2, axis=1)


def _match_length(h, ref):
    length_h = h.shape[1]
    length_ref = ref.shape[1]
    if length_h == length_ref:
        return h
    if length_h > length_ref:
        return h[:, :length_ref, :]
    return jnp.pad(h, [(0, 0), (0, length_ref - length_h), (0, 0)])


def sinusoidal_embedding(t, dim):
    """DDPM sinusoidal embedding of t in [0, 1].

    t is scaled by 1000 before the sinusoid, matching the usual integer
    timestep range. Frequencies use max period 10000.
    """
    t = t * 1000.0
    half = dim // 2
    idx = jnp.arange(half, dtype=t.dtype)
    freqs = jnp.exp(-jnp.log(10000.0) * idx / half)
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, [(0, 0), (0, 1)])
    return emb


def _time_mlp(t, params):
    dim = params["w1"].shape[0]
    emb = sinusoidal_embedding(t, dim)
    hidden = jax.nn.silu(emb @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def _res_block(x, params, temb):
    h = _conv1d(x, params["conv1"])
    h = _group_norm(h, params["gn1"]["scale"], params["gn1"]["bias"])
    h = jax.nn.silu(h)
    t_add = jax.nn.silu(temb) @ params["tproj"]["w"] + params["tproj"]["b"]
    h = h + t_add[:, None, :]
    h = _conv1d(h, params["conv2"])
    h = _group_norm(h, params["gn2"]["scale"], params["gn2"]["bias"])
    h = jax.nn.silu(h)
    if "skip" in params:
        residual = _conv1d(x, params["skip"])
    else:
        residual = x
    return h + residual


def apply(params, x, t, key=None, dropout_rate=0.0):
    """Apply the U-Net.

    x: (batch, length, 9) concatenated cylinder embeddings.
    t: scalar or (batch,) time in [0, 1].
    Returns omega, a_dot, each shaped (batch, length).

    dropout_rate is a scalar. Nonzero dropout needs `key` and is applied
    to the concatenated input only. A rate of 0 leaves the input unchanged.
    """
    x = jnp.asarray(x)
    # Dropout is tracer-safe: rate 0 leaves x unchanged. A missing key is
    # only an error when a Python-side rate is actually nonzero.
    if key is not None:
        rate = jnp.asarray(dropout_rate, dtype=x.dtype)
        keep = jnp.array(1.0, dtype=x.dtype) - rate
        mask = jax.random.bernoulli(key, jnp.clip(keep, 0.0, 1.0), x.shape)
        dropped = x * mask / jnp.maximum(keep, jnp.array(1e-8, dtype=x.dtype))
        x = jnp.where(rate > 0, dropped, x)
    elif dropout_rate:
        raise ValueError("dropout requires a PRNG key")

    t = jnp.asarray(t, dtype=x.dtype)
    if t.ndim == 0:
        t = jnp.full((x.shape[0],), t)
    temb = _time_mlp(t, params["time"])

    h = _conv1d(x, params["stem"])
    skips = []
    for level in params["enc"]:
        for block in level["blocks"]:
            h = _res_block(h, block, temb)
        skips.append(h)
        if "down" in level:
            h = _conv1d(h, level["down"], stride=2)

    for block in params["mid"]["blocks"]:
        h = _res_block(h, block, temb)

    for i, level in enumerate(params["dec"]):
        skip = skips[-(i + 1)]
        h = _match_length(h, skip)
        h = jnp.concatenate([h, skip], axis=-1)
        for block in level["blocks"]:
            h = _res_block(h, block, temb)
        if i < len(params["dec"]) - 1:
            h = _upsample(h)

    out = _conv1d(h, params["head"])
    return out[..., 0], out[..., 1]
