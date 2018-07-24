#ifndef XNOR_MATMUL_H_
#define XNOR_MATMUL_H_

template <typename Device, typename T>
struct XNORmatmulFunctor {
  void operator()(const Device& d, const T* a_mtx, const T* b_mtx, T* out, int m, int n, int k);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct XNORmatmulFunctor {
  void operator()(const Eigen::GpuDevice& d, const T* a_mtx, const T* b_mtx, T* out, int m, int n, int k);
};
#endif

#endif XNOR_MATMUL_H_