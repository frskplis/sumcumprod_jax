# -*- coding: utf-8 -*-

__all__ = ["sumcumprod"]

from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

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

def sumcumprod(input):
    return _sumcumprod_prim.bind(input)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _sumcumprod_abstract(input):
    shape = input.shape
    dtype = dtypes.canonicalize_dtype(input.dtype)
    return (ShapedArray(shape, dtype),)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _sumcumprod_lowering(ctx, input, *, platform="cpu"):

    # Extract the numpy type of the inputs
    input_aval = ctx.avals_in[0]
    np_dtype = np.dtype(input_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(input.type)
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
            out_types=[dtype],
            # The inputs:
            operands=[mlir.ir_constant(size), input],
            # Layout specification:
            operand_layouts=[(), layout],
            result_layouts=[layout]
        )

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
            out_types=[dtype],
            # The inputs:
            operands=[input],
            # Layout specification:
            operand_layouts=[layout],
            result_layouts=[layout],
            # GPU specific additional data
            backend_config=opaque
        )

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
def _sumcumprod_jvp(args, tangents):
    input = args
    d_input = tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    sin_ecc_anom, cos_ecc_anom = _sumcumprod_prim.bind(input, ecc)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    # Propagate the derivatives
    d_ecc_anom = (
        zero_tangent(d_input, input)
        + zero_tangent(d_ecc, ecc) * sin_ecc_anom
    ) / (1 - ecc * cos_ecc_anom)

    return (sin_ecc_anom, cos_ecc_anom), (
        cos_ecc_anom * d_ecc_anom,
        -sin_ecc_anom * d_ecc_anom,
    )


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _sumcumprod_batch(args, axes):
    x, = args
    #bd, = axes
    #x = jnp.moveaxis(x, bd, -1)
    return sumcumprod(x), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_sumcumprod_prim = core.Primitive("sumcumprod")
_sumcumprod_prim.multiple_results = True
_sumcumprod_prim.def_impl(partial(xla.apply_primitive, _sumcumprod_prim))
_sumcumprod_prim.def_abstract_eval(_sumcumprod_abstract)

# Connect the XLA translation rules for JIT compilation
for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _sumcumprod_prim,
        partial(_sumcumprod_lowering, platform=platform),
        platform=platform)

# Connect the JVP and batching rules
ad.primitive_jvps[_sumcumprod_prim] = _sumcumprod_jvp
batching.primitive_batchers[_sumcumprod_prim] = _sumcumprod_batch
