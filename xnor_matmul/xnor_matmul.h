#ifndef XNOR_MATMUL_H_
#define XNOR_MATMUL_H_

template <typename Device, typename T, typename Mask>
struct XnorMatmulFunctor {
  void operator()(const Eigen::ThreadPoolDevice& d, T* a_mtx, T* b_mtx, T* out, Mask* a_msk, Mask* b_msk, int m, int n, int k);
};



#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T, typename Mask>
struct XnorMatmulFunctor<Eigen::GpuDevice, T, Mask> {
  void operator()(const Eigen::GpuDevice& d, T* a_mtx, T* b_mtx, T* out, Mask* a_msk, Mask* b_msk, int m, int n, int k);
};
#endif

#endif