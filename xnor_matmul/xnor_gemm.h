#ifndef XNOR_GEMM_H_
#define XNOR_GEMM_H_

template <typename Device, typename T>
struct XnorGemmFunctor {
  void operator()(const Eigen::ThreadPoolDevice& d, T* a_mtx, T* b_mtx, T* out, int m, int n, int k);
};



#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct XnorGemmFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, T* a_mtx, T* b_mtx, T* out, int m, int n, int k);
};
#endif

#endif