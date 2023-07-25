// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "sumcumprod.h"
#include "pybind11_kernel_helpers.h"

using namespace sumcumprod_jax;

namespace {

template <typename T>
void cpu_sumcumprod(void *out_tuple, const void **in) {
  // Parse the inputs
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const T *mean_anom = reinterpret_cast<const T *>(in[1]);

  // The output is stored as a list of pointers since we have multiple outputs
  void **out = reinterpret_cast<void **>(out_tuple);
  T *sin_ecc_anom = reinterpret_cast<T *>(out[0]);

  for (std::int64_t n = 0; n < size; ++n) {
    compute_eccentric_anomaly(mean_anom[n], sin_ecc_anom + n);
  }
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_sumcumprod_f32"] = EncapsulateFunction(cpu_sumcumprod<float>);
  dict["cpu_sumcumprod_f64"] = EncapsulateFunction(cpu_sumcumprod<double>);
  return dict;
}

PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }

}  // namespace
