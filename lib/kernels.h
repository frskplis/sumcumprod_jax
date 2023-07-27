#ifndef _sumcumprod_JAX_KERNELS_H_
#define _sumcumprod_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace sumcumprod_jax {
struct sumcumprodDescriptor {
  std::int64_t size1;
  std::int64_t size2;
};

void gpu_sumcumprod_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_sumcumprod_f64(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace sumcumprod_jax

#endif