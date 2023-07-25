from jax import vmap, numpy as jnp
from kepler_jax import kepler

def func(x):
  return kepler(x, x)[0]

x = jnp.ones((100, 100))

vmap(func)(x)
