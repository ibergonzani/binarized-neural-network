#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "xnor_matmul.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;


template <typename T>
__device__ unsigned int signArraytoBitmask(T* array)
{
	unsigned int concatenated=0;
    unsigned int sign;
    
    for (int i = 0; i < 32; ++i)
    {
        sign = (array[i] >= 0);				
        concatenated = concatenated | (sign<<i);
    }
    
	return concatenated;
}


template <typename T>
__global__ void concantenateSignMasks(const T* mtx, const unsigned int* sign_mtx, int size)
{
	const unsigned int tid = blockIdx.x * threadDim.x + threadIdx.x;
	
	if(tid < size)
	{
		const unsigned int* sub_mtx = mtx + tid * 32;
		sign_mtx[tid] = signArraytoBitmask(sub_mtx);
	}
	
}


template <typename T>
__global__ void concantenateSignMasks(const T* mtx, const unsigned int* sign_mtx, int size)
{
	const unsigned int tid = blockIdx.x * threadDim.x + threadIdx.x;
	
	if(tid < size)
	{
		const unsigned int* sub_mtx = mtx + tid * 32;
		sign_mtx[tid] = signArraytoBitmask(sub_mtx);
	}
	
}


// Define the CUDA kernel.
template <typename T>
__global__ void matmulCudaKernel(const int size, const T* a_mtx, const T* b_mtx, T* out)
{
	
	
	__shared__ unsigned int Asub[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ unsigned int Bsub[BLOCK_SIZE][BLOCK_SIZE];
	
	
	
	// implementation here
	
	// for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
		// out[i] = 2 * ldg(in + i);
	// }
}


// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void XNORmatmulFunctor<GPUDevice, T>::operator()(const GPUDevice& d, const T* a_mtx, const T* b_mtx, T* out, int m, int n, int k)
{
	
	// allocate memory for concatenated A matrix and b matrix
	unsigned int* ac;
	unsigned int* bc;
	
	cudaMalloc((void**)&ac, m*n*sizeof(unsigned int) / 32)
	cudaMalloc((void**)&bc, n*k*sizeof(unsigned int) / 32)
	
	int thread_per_block = 32
	int block_count = ceil(m * n / 32)
	concantenateSignMasks<T><<<block_count, thread_per_block, 0, d.stream()>>>(a_mtx, ac, m*n/32)
	
	block_count = n * k /32
	concantenateSignMasks<T><<<block_count, thread_per_block, 0, d.stream()>>>(b_mtx, bc, n*k/32)
	
	block_count = 1024;
	thread_per_block = 20;
	matmulCudaKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
	
	cudaFree(bc)
	cudaFree(ac)
}


// Explicitly instantiate functors for the types of OpKernels registered.
template struct XNORmatmulFunctor<GPUDevice, float>;
template struct XNORmatmulFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
