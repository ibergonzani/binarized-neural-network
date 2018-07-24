#ifndef XNOR_MATMUL_H_
#define XNOR_MATMUL_H_

template <typename Device, typename T>
struct XnorMatmulFunctor {
  void operator()(const Eigen::ThreadPoolDevice& d, T* a_mtx, T* b_mtx, T* out, int m, int n, int k);
};



#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct XnorMatmulFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, T* a_mtx, T* b_mtx, T* out, int m, int n, int k);
};
#endif

#endif