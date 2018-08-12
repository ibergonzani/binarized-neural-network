#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cstdio>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "xnor_matmul.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#define BLOCK_SIZE 16


template <typename T>
__device__ __forceinline__ int __popcnt(T x)
{
	return __popc(x);
}

template<>
__device__ __forceinline__ int __popcnt<unsigned long long>(unsigned long long x_ll)
{
	return __popcll(x_ll);
}



template <typename T, typename Mask>
__device__ unsigned int signArraytoBitmask(T* array, int len)
{
	Mask bitmask = 0;
    Mask sign;
    
    for (int i = 0; i < len; ++i)
    {
        sign = (array[i] >= 0);				
        bitmask = bitmask | (sign<<i);
    }
    
	return bitmask;
}


template <typename T, typename To>
__global__ void concantenateRowsSigns(T* mtx, To* sign_mtx, int len)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	int t_size = sizeof(To) * 8;
	
	if(tid < len)
	{
		T* sub_mtx = mtx + tid * t_size;
		sign_mtx[tid] = signArraytoBitmask<T, To>(sub_mtx, t_size);
	}
}



template <typename T, typename To>
__global__ void concantenateRowsSignsShared(T* mtx, To* sign_mtx)
{	
	const int t_size = sizeof(To) * 8;
	const int mult = t_size / 32;
	
	__shared__ To signs[t_size];
	
	int itid = threadIdx.x * mult;
	
	for(int i=0; i < mult; ++i)
		signs[itid + i] = (To)(mtx[blockIdx.x * t_size + itid + i] >= 0);
	
	// syncthreads() not necessary because all threads in a warp (32) execute the same instruction
	// if there is no warp divergence, it is safe also without sync
	
	for(int i = 2; i <=	t_size/2; i = i * 2)
	{
		int curr_id = itid / i * i;
		signs[curr_id] = signs[curr_id] | (signs[curr_id + i/2] << (i/2));	
	}
	
	// here there is warp divergence but it is faster to access signs with one thread and
	// in any case there is no more computation that would require a sync
	if(threadIdx.x == 0)
		sign_mtx[blockIdx.x] = signs[0] | (signs[t_size/2] << t_size/2);
}



template <typename T, typename To>
__global__ void concantenateColumnsSigns(T* mtx, To* sign_mtx, int m, int n, int len)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	int t_size = sizeof(To) * 8;
		
	// in order to avoid other memory alloctions, the sign bitmask generations
	// is done directly in this function (without calling the previous dedicated function)
	
	To bitmask = 0;
	To sign;
	
	if(tid < len)
	{
		for(int i=0; i<t_size; ++i)
		{
			int col = (tid * t_size + i) / m;
			int row = (tid * t_size + i) % m;
			sign = (mtx[row * n + col] >= 0);
			bitmask = bitmask | (sign<<i);
		}
		sign_mtx[tid] = bitmask;
	}
}

template <typename T, typename To>
__global__ void concantenateColumnsSignsShared(T* mtx, To* sign_mtx, int m, int n)
{		
	const int t_size = sizeof(To) * 8;
	const int mult = t_size / 32;
	
	__shared__ To signs[t_size];
	
	const int itid = threadIdx.x * mult;
	
	for(int i=0; i<mult; ++i) 
	{
		int col = (blockIdx.x * t_size + itid + i) / m;
		int row = (blockIdx.x * t_size + itid + i) % m;
		signs[itid + i] = (To)(mtx[row * n + col] >= 0);
	}
	

	for(int i = 2; i <=	t_size/2; i = i * 2)
	{
		int curr_id = itid / i * i;
		signs[curr_id] = signs[curr_id] | (signs[curr_id + i/2] << (i/2));	
	}

	if(threadIdx.x == 0)
		sign_mtx[blockIdx.x] = signs[0] | (signs[t_size/2] << t_size/2);
}


// NAIVE PARALLEL MULTIPLICATION CODE (NOT OPTIMIZED)
// does not use shared memory to speed up memory accesses
// one thread per resultant matrix element
template <typename T, typename Ti>
__global__ void matmulCudaKernelGlobal(Ti* a_mtx, Ti* b_mtx, T* c_mtx, int m, int n, int k)
{
	int trow = blockIdx.y * blockDim.y + threadIdx.y;
	int tcol = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(trow >= m || tcol >= k)
		return;

	int t_size = sizeof(Ti) * 8;
	
	// number of integers to cover an entire row of A or an entire column of B
	int nlen = n / t_size;
	
	Ti* Ar = a_mtx + (trow * nlen);
	Ti* Bc = b_mtx + (tcol * nlen);
	Ti c_value = 0;
	
	for(int i = 0; i < nlen; ++i)
		c_value += __popcnt(Ar[i] ^ Bc[i]);
	
	c_mtx[trow * k + tcol] = -(static_cast<T>(2 * c_value) - n);
}



// OPTIMIZED VERSION WITH SHARED MEMORY
// computes block multiplication copying the blocks data in shared memory
// each block of thread compute a bloc of the resulting matrix
template <typename T, typename Ti>
__global__ void matmulCudaKernelShared(Ti* a_mtx, Ti* b_mtx, T* c_mtx, int m, int n, int k)
{
	
	int block_col = blockIdx.x;
	int block_row = blockIdx.y;
	
	int col = threadIdx.x;
	int row = threadIdx.y;
		
	T* Csub = c_mtx + (block_row * blockDim.y * k + block_col * blockDim.x);
	
	int m_size = sizeof(Ti) * 8;	// number of value in a single mask (32 uint, 64 u long long)
	
	// each thread copy its submatrix element in shared memory
	__shared__ Ti Asub[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ Ti Bsub[BLOCK_SIZE][BLOCK_SIZE];
	
	Ti c_value = 0;
	
	// computing submatrix C_xy associated to the thread block as sum_i(A_xi * B_iy)
	for(int i = 0; i < n / (m_size*BLOCK_SIZE); ++i)
	{
		// getting firt element pointer of submatrix A_xi and B_iy
		Ti* As = a_mtx + (block_row * blockDim.y * n / m_size + blockDim.x * i); 	// a_mtx is a concatenation of rows with 32 elements grouped in one uint
		Ti* Bs = b_mtx + (block_col * blockDim.x * n / m_size + blockDim.y * i);	// b_mtx is a concatenation of columns with 32 elements grouped in one uint
		
		// copy submatrix A_xi and submatrix B_iy to shared memory (each thread one element)
		Asub[row][col] = As[row * n / m_size + col];
		Bsub[row][col] = Bs[col * n / m_size + row];
		
		// evaluating c_value only after all the block threads have copied their part of A and B in shared memory
		__syncthreads();
		
		for(int j=0; j < BLOCK_SIZE; ++j)
			c_value += __popcnt((Asub[row][j] ^ Bsub[j][col]));
		
		__syncthreads();
	}	
	
	Csub[row * k + col] = -(static_cast<T>(2 * c_value) - n);
}




// Define the GPU implementation that launches the CUDA kernel.
template <typename T, typename mask_t>
void XnorMatmulFunctor<GPUDevice, T, mask_t>::operator()(const GPUDevice& d, T* a_mtx, T* b_mtx, T* out, mask_t* ac, mask_t* bc, int m, int n, int k)
{
	int mask_size = sizeof(mask_t) * 8;	
		
	int thread_per_block = 32;	
	int block_count = (m * n) / mask_size;
	concantenateRowsSignsShared<T, mask_t> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac);
	
	// block_count = (m * n) / (thread_per_block * mask_size) + 1;
	// concantenateRowsSigns<T, mask_t> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac, (m*n)/mask_size);
	
	block_count = (n * k) / (thread_per_block * mask_size) + 1;
	concantenateColumnsSigns<T, mask_t> <<<block_count, thread_per_block, 0, d.stream()>>> (b_mtx, bc, n, k, (n*k)/mask_size);
	
	block_count = (n * k) / mask_size;
	concantenateColumnsSignsShared<T, mask_t> <<<block_count, thread_per_block, 0, d.stream()>>> (b_mtx, bc, n, k);
	
	
	dim3 block_dims(k/BLOCK_SIZE, m/BLOCK_SIZE);
	dim3 thread_dims(BLOCK_SIZE, BLOCK_SIZE);
	// matmulCudaKernelGlobalT<T, mask_t><<<block_dims, thread_dims, 0, d.stream()>>>(ac, bc, out, m, n, k);
	matmulCudaKernelShared<T, mask_t><<<block_dims, thread_dims, 0, d.stream()>>>(ac, bc, out, m, n, k);

}


// Explicitly instantiate functors for the types of OpKernels registered.
template struct XnorMatmulFunctor<GPUDevice, float, unsigned int>;
template struct XnorMatmulFunctor<GPUDevice, int32, unsigned int>;
template struct XnorMatmulFunctor<GPUDevice, float, unsigned long long>;
template struct XnorMatmulFunctor<GPUDevice, int32, unsigned long long>;

#endif  // GOOGLE_CUDA
