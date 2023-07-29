# Extending JAX with custom C++ and CUDA code
This is sumcumprod extension for CUDA and CPU in C++ for JAX that is equivalent to this python code:

```
def grand_true_function(x,y):
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape
    i = jnp.arange(x.shape[0])
    mask = i[None, :] < i[:, None]
    cumprod = jnp.where(mask, 1, 1 / (1 + x[None, :] * y[:, None])).cumprod(1)
    return jnp.where(mask, 0, cumprod).sum(1)
```

but should be much faster in execution due to smaller memory requirements.