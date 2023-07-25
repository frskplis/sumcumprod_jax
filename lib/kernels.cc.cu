// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "kepler.h"
#include "kernel_helpers.h"
#include "kernels.h"

namespace kepler_jax {

namespace {

template <typename T>
__global__ void kepler_kernel(std::int64_t size, const T *mean_anom, const T *ecc, T *sin_ecc_anom) {

  float total_sum = 0.0;
  float multi = 1.0;

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (int i = tid; i < size; i++){
	multi = multi * mean_anom[i];
	total_sum += multi;
  }
  sin_ecc_anom[tid] = total_sum;

}

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
inline void apply_kepler(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
  const KeplerDescriptor &d = *UnpackDescriptor<KeplerDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;

  const T *mean_anom = reinterpret_cast<const T *>(buffers[0]);
  const T *ecc = reinterpret_cast<const T *>(buffers[1]);
  T *sin_ecc_anom = reinterpret_cast<T *>(buffers[2]);

  const int block_dim = 32;
  int NUM_THREADS = 64;  
  int NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;

const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);
  kepler_kernel<T>
      <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(size, mean_anom, ecc, sin_ecc_anom);

  ThrowIfError(cudaGetLastError());
}

}  // namespace

void gpu_kepler_f32(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_kepler<float>(stream, buffers, opaque, opaque_len);
}

void gpu_kepler_f64(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_kepler<double>(stream, buffers, opaque, opaque_len);
}

}  // namespace kepler_jax
