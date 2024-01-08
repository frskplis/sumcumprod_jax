# -*- coding: utf-8 -*-

__all__ = ["sumcumprod"]

from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
import jax

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")   

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

# This function exposes the primitive to user code and this is the only
# public-facing function in this module

def sumcumprod(input1, input2):
    x, y = jnp.broadcast_arrays(input1, input2)
    return _sumcumprod_prim.bind(x, y)

def sumcumprod_masked(input1, input2):
    x, y = jnp.broadcast_arrays(input1, input2)
    return _sumcumprod_masked_prim.bind(x, y)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _sumcumprod_abstract(input1, input2):
    assert input1.shape == input2.shape, f"shapes mismtach {input1.shape=}, {input2.shape=}"
    shape = input1.shape
    dtype = dtypes.canonicalize_dtype(input1.dtype)
    return ShapedArray(shape, dtype)

def _sumcumprod_masked_abstract(input1, input2):
    assert input1.shape == input2.shape, f"shapes mismtach {input1.shape=}, {input2.shape=}"
    shape = input1.shape
    dtype = dtypes.canonicalize_dtype(input1.dtype)
    return ShapedArray(shape, dtype)

# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _sumcumprod_lowering(ctx, input1, input2, *, platform="cpu"):

    assert input1.type == input2.type, f"Mismatched types {input1.type} and {input2.type}"

    # Extract the numpy type of the inputs
    input1_aval, _ = ctx.avals_in
    np_dtype = np.dtype(input1_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(input1.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)
    int_size_of_last_dim = dims[-1]

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_sumcumprod_f32"
    elif np_dtype == np.float64:
        op_name = platform + "_sumcumprod_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return custom_call(
            op_name,
            # Output types
            result_types=[dtype],
            # The inputs have to be int32 because for int64 for some reason it does not work:
            operands=[mlir.ir_constant(np.int32(size)), mlir.ir_constant(np.int32(int_size_of_last_dim)), input1, input2],
            # Layout specification:
            operand_layouts=[(), (), layout, layout],
            result_layouts=[layout]
        ).results

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'sumcumprod_jax' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_sumcumprod_descriptor(size, int_size_of_last_dim)

        return custom_call(
            op_name,
            # Output types
            result_types=[dtype],
            # The inputs:
            operands=[input1, input2],
            # Layout specification:
            operand_layouts=[layout, layout],
            result_layouts=[layout],
            # GPU specific additional data
            backend_config=opaque
        ).results

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )

def _sumcumprod_masked_lowering(ctx, input1, input2, *, platform="cpu"):

    assert input1.type == input2.type, f"Mismatched types {input1.type} and {input2.type}"

    # Extract the numpy type of the inputs
    input1_aval, _ = ctx.avals_in
    np_dtype = np.dtype(input1_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(input1.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)
    int_size_of_last_dim = dims[-1]

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_sumcumprod_masked_f32"
    elif np_dtype == np.float64:
        op_name = platform + "_sumcumprod_masked_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    print(int_size_of_last_dim)
    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return custom_call(
            op_name,
            # Output types
            result_types=[dtype],
            # The inputs have to be int32 because for int64 for some reason it does not work:
            operands=[mlir.ir_constant(np.int32(size)), mlir.ir_constant(np.int32(int_size_of_last_dim)), input1, input2],
            # Layout specification:
            operand_layouts=[(), (), layout, layout],
            result_layouts=[layout]
        ).results

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'sumcumprod_jax' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_sumcumprod_descriptor(size, int_size_of_last_dim)

        return custom_call(
            op_name,
            # Output types
            result_types=[dtype],
            # The inputs:
            operands=[input1, input2],
            # Layout specification:
            operand_layouts=[layout, layout],
            result_layouts=[layout],
            # GPU specific additional data
            backend_config=opaque
        ).results

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# Here we define the differentiation rules using a JVP derived using implicit
# differentiation of Kepler's equation:
#
#  M = E - e * sin(E)
#  -> dM = dE * (1 - e * cos(E)) - de * sin(E)
#  -> dE/dM = 1 / (1 - e * cos(E))  and  de/dM = sin(E) / (1 - e * cos(E))
#
# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.

def _pure_jax_sumcumprod(x,y):
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape
    i = jnp.arange(x.shape[0])
    mask = i[None, :] < i[:, None]
    cumprod = jnp.where(mask, 1, 1 / (1 + x[None, :] * y[:, None])).cumprod(1)
    return jnp.where(mask, 0, cumprod).sum(1)

def _sumcumprod_jvp(args, tangents):
    # Here we use jax pure function to calculate primals and tangents
    input1, input2 = args
    d_input1, d_input2 = tangents

    res = jax.jvp(jax.vmap(_pure_jax_sumcumprod), (input1, input2, ), (d_input1, d_input2, ))

    return res

def _sumcumprod_masked_jvp(args, tangents):
    # Here we use jax pure function to calculate primals and tangents
    input1, input2 = args
    d_input1, d_input2 = tangents

    res = jax.jvp(jax.vmap(_pure_jax_sumcumprod), (input1, input2, ), (d_input1, d_input2, ))

    return res


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _sumcumprod_batch(args, axes):
    #assert axes[0] == axes[1], f"Incorrect dimensions axes[0] = {axes[0]}, axes[1] = {axes[1]}"
    x, y = args[0], args[1]
    # bd1, bd2 = axes[0], axes[1]
    # x = jnp.moveaxis(x, bd1, 0)
    # y = jnp.moveaxis(y, bd2, 0)

    return sumcumprod(x,y), 0


def _sumcumprod_masked_batch(args, axes):
    #assert axes[0] == axes[1], f"Incorrect dimensions axes[0] = {axes[0]}, axes[1] = {axes[1]}"
    x, y = args[0], args[1]
    # bd1, bd2 = axes[0], axes[1]
    # x = jnp.moveaxis(x, bd1, 0)
    # y = jnp.moveaxis(y, bd2, 0)

    return sumcumprod_masked(x,y), 0




# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_sumcumprod_prim = core.Primitive("sumcumprod")
_sumcumprod_prim.multiple_results = False
_sumcumprod_prim.def_impl(partial(xla.apply_primitive, _sumcumprod_prim))
_sumcumprod_prim.def_abstract_eval(_sumcumprod_abstract)

_sumcumprod_masked_prim = core.Primitive("sumcumprod")
_sumcumprod_masked_prim.multiple_results = False
_sumcumprod_masked_prim.def_impl(partial(xla.apply_primitive, _sumcumprod_masked_prim))
_sumcumprod_masked_prim.def_abstract_eval(_sumcumprod_masked_abstract)


# Connect the XLA translation rules for JIT compilation
for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _sumcumprod_prim,
        partial(_sumcumprod_lowering, platform=platform),
        platform=platform)
    
    mlir.register_lowering(
        _sumcumprod_masked_prim,
        partial(_sumcumprod_masked_lowering, platform=platform),
        platform=platform)

# Connect the JVP and batching rules
ad.primitive_jvps[_sumcumprod_prim] = _sumcumprod_jvp
batching.primitive_batchers[_sumcumprod_prim] = _sumcumprod_batch

ad.primitive_jvps[_sumcumprod_masked_prim] = _sumcumprod_masked_jvp # todo: correct this
batching.primitive_batchers[_sumcumprod_masked_prim] = _sumcumprod_masked_batch # todo: correct this


