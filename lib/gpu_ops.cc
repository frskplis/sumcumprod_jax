// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_sumcumprod_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace sumcumprod_jax;

namespace {
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_sumcumprod_f32"] = EncapsulateFunction(gpu_sumcumprod_f32);
  dict["gpu_sumcumprod_f64"] = EncapsulateFunction(gpu_sumcumprod_f64);
  dict["gpu_sumcumprod_masked_f32"] = EncapsulateFunction(gpu_sumcumprod_masked_f32);
  dict["gpu_sumcumprod_masked_f64"] = EncapsulateFunction(gpu_sumcumprod_masked_f64);

  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("registrations", &Registrations);
  m.def("build_sumcumprod_descriptor",
        [](std::int64_t size1, std::int64_t size2) { return PackDescriptor(sumcumprodDescriptor{size1, size2}); });
}
}  // namespace
