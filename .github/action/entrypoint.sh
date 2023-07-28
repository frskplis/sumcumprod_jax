#!/bin/sh -l

cd /github/workspace
KEPLER_JAX_CUDA=yes python3 -m pip install .
python3 -c 'import sumcumprod_jax;print(sumcumprod_jax.__version__)'
python3 -c 'import sumcumprod_jax.gpu_ops'
