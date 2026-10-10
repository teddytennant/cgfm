Cylindrical geodesic flow matching is a JAX implementation of the phase-amplitude cylinder bridge from arXiv:2610.08510.

pip install -e .
python3 -m pytest

Tests check the closed-form geodesic, the constant-norm oracle field, the affine-bridge distortion bound, Hilbert amplitude, Heun projection, and a tiny flow-matching step on synthetic pairs. They do not train the 1M-parameter U-Net, do not use PPG or SCG recordings, and do not implement the extended Hilbert transform.
