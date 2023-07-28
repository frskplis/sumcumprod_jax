# -*- coding: utf-8 -*-

import numpy as np
import pytest

import jax
from jax import numpy as jnp, vmap
from jax.config import config
from jax.test_util import check_grads

from sumcumprod_jax import sumcumprod

config.update("jax_enable_x64", True)

def grand_true_function(x,y):
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape
    i = jnp.arange(x.shape[0])
    mask = i[None, :] < i[:, None]
    cumprod = jnp.where(mask, 1, 1 / (1 + x[None, :] * y[:, None])).cumprod(1)
    return jnp.where(mask, 0, cumprod).sum(1)

def vectorized_grand_true_function(x,y):
    return jax.vmap(grand_true_function)(x,y)

@pytest.fixture()
def sumcumprod_data():
    # Note about precision: the precision of the mod function in float32 is not
    # great so we're only going to test values in the range ~0-2*pi. In real
    # world applications, the mod should be done in float64 even if the solve
    # is done in float32
    # ecc = np.linspace(0, 0.9, 55)
    # true_ecc_anom = np.linspace(-np.pi, np.pi, 101)
    # mean_anom = true_ecc_anom - ecc[:, None] * np.sin(true_ecc_anom)
    # dtype = request.param

    x = np.ones((10,10))
    y = np.ones((10,10))

    return (
        x, y, vectorized_grand_true_function(x,y),
    )

def test_check_sumcumprod(sumcumprod_data):
    x, y, grand_true_result = sumcumprod_data
    np.testing.assert_allclose(sumcumprod(x,y), grand_true_result, atol=1e-5)

def test_check_jit(sumcumprod_data):
    x, y, grand_true_result = sumcumprod_data
    result = jax.jit(sumcumprod)(x,y)
    np.testing.assert_allclose(result, grand_true_result, atol=1e-5)

def test_check_vmap(sumcumprod_data):
    x, y, grand_true_result = sumcumprod_data
    result = vmap(sumcumprod)(x,y)
    np.testing.assert_allclose(result, grand_true_result, atol=1e-5)

def test_check_jvp(sumcumprod_data):
    x, y, grand_true_result = sumcumprod_data
    np.testing.assert_allclose(jax.jvp(sumcumprod, (x,y, ), (x, y, )), jax.jvp(vectorized_grand_true_function, (x, y, ), (x, y, )))