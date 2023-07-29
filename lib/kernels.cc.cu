// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "kernel_helpers.h"
#include "kernels.h"

namespace sumcumprod_jax {

namespace {

template <typename T>
__global__ void sumcumprod_kernel(std::int64_t size, int size_of_last_dim, const T *input_array, const T *rs, T *output_array) {

  int tid_start = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid_end = tid_start + (size_of_last_dim - tid_start%size_of_last_dim);

  float total_sum = 0.0;
  float multi_prev = 1.0;
  float multi_cur = 0.0;

	  for (int i = tid_start; i < tid_end; i++){
      multi_cur = __fdividef(1., (1. + input_array[i] * rs[tid_start]));
      multi_cur = multi_prev * multi_cur;
      multi_prev = multi_cur;
      total_sum += multi_cur;
	  }
	  output_array[tid_start] = total_sum;

}

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
inline void apply_sumcumprod(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
  const sumcumprodDescriptor &d = *UnpackDescriptor<sumcumprodDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size1;
  const std::int64_t size_of_last_dim = d.size2;

  const T *input_array1 = reinterpret_cast<const T *>(buffers[0]);
  const T *input_array2 = reinterpret_cast<const T *>(buffers[1]);
  T *output_array = reinterpret_cast<T *>(buffers[2]);

  int NUM_THREADS = 64;  
	int NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;

  sumcumprod_kernel<T>
      <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(size, size_of_last_dim, input_array1, input_array2, output_array);

  ThrowIfError(cudaGetLastError());
}

}  // namespace

void gpu_sumcumprod_f32(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_sumcumprod<float>(stream, buffers, opaque, opaque_len);
}

void gpu_sumcumprod_f64(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_sumcumprod<double>(stream, buffers, opaque, opaque_len);
}

}  // namespace sumcumprod_jax
