"""Standard FFT Hilbert transform.

The extended Hilbert transform (EHT) is not implemented.
"""

import jax.numpy as jnp


def hilbert_multiplier(length, dtype=jnp.float32):
    """Frequency response of the analytic-signal Hilbert transformer.

    Even T (odd T is analogous):
    H_0 = 1, H_{T/2} = 1 if T is even,
    H_k = 2 for 1 <= k < T/2, and H_k = 0 otherwise.
    """
    length = int(length)
    h = jnp.zeros((length,), dtype=dtype)
    h = h.at[0].set(1.0)
    if length % 2 == 0:
        h = h.at[length // 2].set(1.0)
        h = h.at[1 : length // 2].set(2.0)
    else:
        h = h.at[1 : (length + 1) // 2].set(2.0)
    return h


def analytic_signal(x):
    """z = ifft(H * fft(x)) along the last axis.

    For a real input the real part of z equals x up to FFT roundoff.
    """
    x = jnp.asarray(x)
    spectrum = jnp.fft.fft(x, axis=-1)
    h = hilbert_multiplier(x.shape[-1], dtype=jnp.float32)
    # Match the computation dtype of the spectrum (float32 -> complex64).
    h = h.astype(spectrum.real.dtype)
    return jnp.fft.ifft(spectrum * h, axis=-1)


def amplitude_phase(x, amp_floor=1e-12):
    """Amplitude and unwrapped phase of the analytic signal.

    a = max(|z|, amp_floor), phi = unwrap(angle(z)).
    """
    z = analytic_signal(x)
    amplitude = jnp.maximum(jnp.abs(z), jnp.array(amp_floor, dtype=jnp.abs(z).dtype))
    phase = jnp.unwrap(jnp.angle(z), axis=-1)
    return amplitude, phase


def waveform_to_embedding(x, amp_floor=1e-12):
    """Hilbert embedding (cos phi, sin phi, a) of a real waveform."""
    from cgfm.geometry import embed

    amplitude, phase = amplitude_phase(x, amp_floor=amp_floor)
    return embed(phase, amplitude), amplitude, phase
