// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include <omp.h>
#include "pybind11_kernel_helpers.h"

using namespace sumcumprod_jax;

namespace {

template <typename T>
void cpu_sumcumprod(void *out, const void **in) {
  // Parse the inputs
  const std::int32_t size = *reinterpret_cast<const std::int32_t *>(in[0]);
  const std::int32_t size_of_last_dim = *reinterpret_cast<const std::int32_t *>(in[1]);

  const T *input1 = reinterpret_cast<const T *>(in[2]);
  const T *input2 = reinterpret_cast<const T *>(in[3]);

  // The output is stored as a list of pointers since we have multiple outputs
  //void **out = reinterpret_cast<void **>(out_tuple);
  T *output = reinterpret_cast<T *>(out);
  
  #pragma omp parallel for
  for (std::int32_t i = 0; i < size; i++){
    float total_sum = 0.0;
    float multi_prev = 1.0;
    float multi_cur = 0.0;
    std::int32_t idx_end = i + (size_of_last_dim - i%size_of_last_dim);

      for (std::int32_t j = i; j < idx_end; j++) {
      multi_cur = 1. / (1. + input1[i] * input2[i]);
      multi_cur = multi_prev * multi_cur;
      multi_prev = multi_cur;
      total_sum += multi_cur;
	  }
	  output[i] = total_sum;
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
