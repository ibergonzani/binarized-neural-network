#ifndef XNOR_MATMUL_H_
#define XNOR_MATMUL_H_

template <typename Device, typename T>
struct XNORmatmulFunctor {
  void operator()(const Eigen::ThreadPoolDevice& d, const T* a_mtx, const T* b_mtx, T* out, int m, int n, int k);
};



#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct XNORmatmulFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const T* a_mtx, const T* b_mtx, T* out, int m, int n, int k);
};
#endif

#endif