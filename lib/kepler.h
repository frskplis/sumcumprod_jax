// This header defines the actual algorithm for our op. It is reused in cpu_ops.cc and
// kernels.cc.cu to expose this as a XLA custom call. The details aren't too important
// except that directly implementing this algorithm as a higher-level JAX function
// probably wouldn't be very efficient. That being said, this is not meant as a
// particularly efficient or robust implementation. It's just here to demonstrate the
// infrastructure required to extend JAX.

#ifndef _KEPLER_JAX_KEPLER_H_
#define _KEPLER_JAX_KEPLER_H_

#include <cmath>

namespace kepler_jax {

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifdef __CUDACC__
#define KEPLER_JAX_INLINE_OR_DEVICE __host__ __device__
#else
#define KEPLER_JAX_INLINE_OR_DEVICE inline

template <typename T>
inline void sincos(const T& x, T* sx, T* cx) {
  *sx = sin(x);
  *cx = cos(x);
}
#endif

template <typename T>
KEPLER_JAX_INLINE_OR_DEVICE void compute_eccentric_anomaly(const T& mean_anom, const T& ecc,
                                                           T* sin_ecc_anom) {
  *sin_ecc_anom = 1.0;
}

}  // namespace kepler_jax

#endif
