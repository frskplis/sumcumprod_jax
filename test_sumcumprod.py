# # %%
# from jax import vmap, numpy as jnp
# from jax import jit
# from sumcumprod_jax import sumcumprod
# from jax import make_jaxpr

# def my_kernel(x):
#   return sumcumprod(x)

# x = jnp.ones((10000, 100)) * 0.25
# y = jnp.ones((10000, 100)) * 0.05

# # def pure_jax(x):
# #   i = jnp.arange(x.shape[0])
# #   mask = i[None, :] < i[:, None]
# #   cumprod = jnp.where(mask, 1, x[None, :]).cumprod(1)
# #   return jnp.where(mask, 0, cumprod).sum(1)

# # def func1(x):
# #   my_lst = []
# #   for i in range(x.shape[0]):
# #     res = jnp.sum(jnp.cumprod(x[i:]))
# #     my_lst.append(res)
# #   return jnp.stack(my_lst)

# # def func2(x):
# #   my_lst = []
# #   for i in range(x.shape[0]):
# #     s = 0.0
# #     p = 1.0
# #     for a in x[i:]:
# #       p*=a
# #       s+=p
# #     my_lst.append(s)
# #   return jnp.stack(my_lst)

# # def func2_xy(x, y):
# #   my_lst = []
# #   for i in range(x.shape[0]):
# #     s = 0.0
# #     p = 1.0
# #     for a in x[i:]:
# #       p*=a * y
# #       s+=p
# #     my_lst.append(s)
# #   return jnp.stack(my_lst)

# # func2_vmap_jit = jit(vmap(func2))
# # my_kernel_vmap_jit  = jit(vmap(my_kernel))
# # pure_jax_vmap_jit  = jit(vmap(pure_jax))

# # def myfunc(x, y):
# #   my_lst = []
# #   for i in range(x.shape[0]):
# #     res = jnp.sum(jnp.cumprod(x[i:] * y[i]))
# #     my_lst.append(res)
# #   return jnp.stack(my_lst)

# # def myfunc_cuda(x, y):
# #   my_lst = []
# #   for i in range(x.shape[0]):
# #     res = sumcumprod(x)[0]
# #     my_lst.append(res)
# #   return jnp.stack(my_lst)

# # myfunc_vmap_jit = jit(vmap(myfunc))
# # myfunc_cuda_vmap_jit = jit(vmap(myfunc_cuda))

# from jax import jit, vmap, numpy as jnp
# import numpy as np

# def func_faster_slow_compile(x, y):
#   my_lst = []
#   for i in range(x.shape[0]):
#     s = 0.0
#     p = 1.0
#     for a in x[i:]:
#       p*=(1 / (1 + a * y[i]))
#       s+=p
#     my_lst.append(s)
#   return jnp.stack(my_lst)

# def func1(x, y):
#   my_lst = []
#   for i in range(x.shape[0]):
#     res = jnp.sum(jnp.cumprod(1 / (1 + x[i:] * y[i])))
#     my_lst.append(res)
#   return jnp.stack(my_lst)

# def myfunc(x, y):
#   return sumcumprod(x, y)[0]

# func_faster_slow_compile_vmap_jit = jit(vmap(func_faster_slow_compile))
# func1_vmap_jit = jit(vmap(func1))
# my_kernel_vmap_jit = jit(sumcumprod)

# #sumcumprod(x)

# %%
from jax import numpy as jnp, vmap
from sumcumprod_jax import sumcumprod

x = jnp.ones((10,10))
y = jnp.ones((10,10))

vmap(sumcumprod)(x,y)
# # %%

# %%