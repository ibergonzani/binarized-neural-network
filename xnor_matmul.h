#ifndef XNOR_MATMUL_H_
#define XNOR_MATMUL_H_

template <typename Device, typename T>
struct XNORmatmulFunctor {
  void operator()(const Device& d, int size, const T* a_mtx, const T* b_mtx, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct XNORmatmulFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* a_mtx, const T* b_mtx, T* out);
};
#endif

#endif XNOR_MATMUL_H_