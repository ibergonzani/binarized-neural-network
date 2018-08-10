#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

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


// template <typename T>
// __global__ void concantenateRowsSigns(T* mtx, unsigned int* sign_mtx, int size)
// {
	// const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// if(tid < size)
	// {
		// T* sub_mtx = mtx + tid * 32;
		// sign_mtx[tid] = signArraytoBitmask<T, unsigned int>(sub_mtx, 32);
	// }
	
// }

// template <typename T>
// __global__ void concantenateRowsSigns_ll(T* mtx, unsigned long long* sign_mtx, int size)
// {
	// const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// if(tid < size)
	// {
		// T* sub_mtx = mtx + tid * 64;
		// sign_mtx[tid] = signArraytoBitmask<T, unsigned long long>(sub_mtx, 64);
	// }
	
// }

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


// template <typename T>
// __global__ void concantenateRowsSignsShared(T* mtx, unsigned int* sign_mtx)
// {
	// __shared__ unsigned int signs[32];
	
	// signs[threadIdx.x] = (mtx[blockIdx.x * blockDim.x + threadIdx.x] >= 0);
	
	// // syncthreads() not necessary because all threads in a warp (32) execute the same instruction
	// // if there is no warp divergence, it is safe also without sync
	
	// for(int i = 2; i <=	16; i = i * 2)
	// {
		// int curr_id = threadIdx.x / i * i;
		// signs[curr_id] = signs[curr_id] | (signs[curr_id + i/2] << (i/2));	
	// }
	
	// // here there is warp divergence but it is faster to access signs with one thread and
	// // in any case there is no more computation that would require a sync
	// if(threadIdx.x == 0)
		// sign_mtx[blockIdx.x] = signs[0] | (signs[16] << 16);
// }


template <typename T, typename To>
__global__ void concantenateRowsSignsShared(T* mtx, To* sign_mtx)
{	
	const int wlen = sizeof(To) * 8;
	const int mult = wlen / 32;
	
	__shared__ To signs[wlen];
	
	int itid = threadIdx.x * mult;
	
	for(int i=0; i < mult; ++i)
		signs[itid + i] = (mtx[blockIdx.x * blockDim.x + itid + i] >= 0);
	
	// syncthreads() not necessary because all threads in a warp (32) execute the same instruction
	// if there is no warp divergence, it is safe also without sync
	
	for(int i = 2; i <=	wlen/2; i = i * 2)
	{
		int curr_id = itid / i * i;
		signs[curr_id] = signs[curr_id] | (signs[curr_id + i/2] << (i/2));	
	}
	
	// here there is warp divergence but it is faster to access signs with one thread and
	// in any case there is no more computation that would require a sync
	if(threadIdx.x == 0)
		sign_mtx[blockIdx.x] = signs[0] | (signs[wlen/2] << wlen/2);
}



// template <typename T>
// __global__ void concantenateColumnsSigns(T* mtx, unsigned int* sign_mtx, int m, int n, int size)
// {
	// const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// // in order to avoid other memory alloctions, the sign bitmask generations
	// // is done directly in this function (without calling the previous dedicated function)
	
	// unsigned int bitmask = 0;
	// unsigned int sign;
	
	// if(tid < size)
	// {
		// for(int i=0; i<32; ++i)
		// {
			// int col = (tid*32 + i) / m;
			// int row = (tid*32 + i) % m;
			// sign = (mtx[row * n + col] >= 0);
			// bitmask = bitmask | (sign<<i);
		// }
		// sign_mtx[tid] = bitmask;
	// }
// }

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


// NAIVE PARALLEL MULTIPLICATION CODE (NOT OPTIMIZED)
// does not use shared memory to speed up memory accesses
// one thread per resultant matrix element

// template <typename T>
// __global__ void matmulCudaKernelGlobal(unsigned int* a_mtx, unsigned int* b_mtx, T* c_mtx, int m, int n, int k)
// {
	
	// int trow = blockIdx.y * blockDim.y + threadIdx.y;
	// int tcol = blockIdx.x * blockDim.x + threadIdx.x;
	
	// if(trow >= m || tcol >= k)
		// return;

	// // number of integers to cover an entire row of A or an entire column of B
	// int nlen = n / 32;
	
	// unsigned int* Ar = a_mtx + (trow * nlen);
	// unsigned int* Bc = b_mtx + (tcol * nlen);
	// unsigned int c_value = 0;
	
	// for(int i = 0; i < nlen; ++i)
		// c_value += __popc(Ar[i] ^ Bc[i]);
	
	// c_mtx[trow * k + tcol] = -(static_cast<T>(2 * c_value) - n);
// }


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
// template <typename T>
// __global__ void matmulCudaKernelShared(unsigned int* a_mtx, unsigned int* b_mtx, T* c_mtx, int m, int n, int k)
// {
	
	// int block_col = blockIdx.x;
	// int block_row = blockIdx.y;
	
	// int col = threadIdx.x;
	// int row = threadIdx.y;
		
	// T* Csub = c_mtx + (block_row * blockDim.y * k + block_col * blockDim.x);
	
	// // each thread copy its submatrix data in in shared memory
	// // each A submatrix block is BLOCK_SIZE*(32*BLOCK_SIZE) --> hence BLOCK_SIZE*BLOCK_SIZE wide unsigned int (32 bits) matrix
	// // each B submatrix block is (32*BLOCK_SIZE)*BLOCK_SIZE --> hence BLOCK_SIZE*BLOCK_SIZE wide unsigned int (32 bits) matrix
	// __shared__ unsigned int Asub[BLOCK_SIZE][BLOCK_SIZE];
	// __shared__ unsigned int Bsub[BLOCK_SIZE][BLOCK_SIZE];
	
	// unsigned int c_value = 0;
	
	// // computing submatrix C_xy associated to the thread block as sum_i(A_xi * B_iy)
	// for(int i = 0; i < n / (32*BLOCK_SIZE); ++i)
	// {
		// // getting firt element pointer of submatrix A_xi and B_iy
		// unsigned int* As = a_mtx + (block_row * blockDim.y * n / 32 + blockDim.x * i); 	// a_mtx is a concatenation of rows with 32 elements grouped in one uint
		// unsigned int* Bs = b_mtx + (block_col * blockDim.x * n / 32 + blockDim.y * i);	// b_mtx is a concatenation of columns with 32 elements grouped in one uint
		
		// // copy submatrix A_xi and submatrix B_iy to shared memory (each thread one element)
		// Asub[row][col] = As[row * n / 32 + col];
		// Bsub[row][col] = Bs[col * n / 32 + row];
		
		// // evaluating c_value only after all the block threads have copied their part of A and B in shared memory
		// __syncthreads();
		
		// for(int j=0; j < BLOCK_SIZE; ++j)
			// c_value += __popc((Asub[row][j] ^ Bsub[j][col]));
		
		// __syncthreads();
	// }	
	
	// Csub[row * k + col] = -(static_cast<T>(2 * c_value) - n); //c_value - (n - c_value);
	// // c_mtx[trow * k + tcol] = -(static_cast<T>(2 * c_value) - n);
// }

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
	
	// each thread copy its submatrix data in in shared memory
	// each A submatrix block is BLOCK_SIZE*(32*BLOCK_SIZE) --> hence BLOCK_SIZE*BLOCK_SIZE wide unsigned int (32 bits) matrix
	// each B submatrix block is (32*BLOCK_SIZE)*BLOCK_SIZE --> hence BLOCK_SIZE*BLOCK_SIZE wide unsigned int (32 bits) matrix
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
			c_value += __popc((Asub[row][j] ^ Bsub[j][col]));
		
		__syncthreads();
	}	
	
	Csub[row * k + col] = -(static_cast<T>(2 * c_value) - n); //c_value - (n - c_value);
	// c_mtx[trow * k + tcol] = -(static_cast<T>(2 * c_value) - n);
}

#include <cstdio>

// Define the GPU implementation that launches the CUDA kernel.
template <typename T, typename mask_t>
void XnorMatmulFunctor<GPUDevice, T, mask_t>::operator()(const GPUDevice& d, T* a_mtx, T* b_mtx, T* out, mask_t* ac, mask_t* bc, int m, int n, int k)
{
	
	// // allocate memory for concatenated A matrix and b matrix
	// unsigned int* ac;
	// unsigned int* bc;
	// unsigned int* ac2;
	// unsigned long long* ac64;
	// unsigned long long* bc64;
	
	// cudaMalloc((void**)&ac, m*n*sizeof(unsigned int) / 32);
	// cudaMalloc((void**)&bc, n*k*sizeof(unsigned int) / 32);
	// cudaMalloc((void**)&ac2, m*n*sizeof(unsigned int) / 32);
	// cudaMalloc((void**)&ac64, m*n*sizeof(unsigned long long) / 64);
	// cudaMalloc((void**)&bc64, n*k*sizeof(unsigned long long) / 64);
	
	// printf("SIZEOF HOST %ld\n", sizeof(unsigned int));
	
	int mask_size = sizeof(mask_t) * 8;
	
		
	int thread_per_block = 32;
	int block_count = (m * n) / (thread_per_block * mask_size) + 1;
	// concantenateRowsSigns<T> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac, (m*n)/32);
	// concantenateRowsSigns<T, unsigned int> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac, (m*n)/32);
	
	block_count = (m*n) / mask_size;
	// concantenateRowsSignsShared<T> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac2);
	concantenateRowsSignsShared<T, mask_t> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac);
	
	// block_count = (m*n)/64;
	// concantenateRowsSignsSharedT<T, unsigned long long> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac64);
	
	// block_count = (m * n) / (thread_per_block * 64) + 1;
	// concantenateRowsSigns_ll<T> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac64, (m*n)/64);
	// concantenateRowsSigns<T, unsigned long long> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac64, (m*n)/64);
	
	block_count = (n * k) / (thread_per_block * mask_size) + 1;
	// concantenateColumnsSigns<T> <<<block_count, thread_per_block, 0, d.stream()>>> (b_mtx, bc, n, k, (n*k)/32);
	concantenateColumnsSigns<T, mask_t> <<<block_count, thread_per_block, 0, d.stream()>>> (b_mtx, bc, n, k, (n*k)/32);
	
	// block_count = (n * k) / (thread_per_block * 64) + 1;
	// concantenateColumnsSigns<T, unsigned long long> <<<block_count, thread_per_block, 0, d.stream()>>> (b_mtx, bc64, n, k, (n*k)/64);
	
	
	
	dim3 block_dims(k/BLOCK_SIZE, m/BLOCK_SIZE);
	dim3 thread_dims(BLOCK_SIZE, BLOCK_SIZE);
	// matmulCudaKernelGlobal<T><<<block_dims, thread_dims, 0, d.stream()>>>(ac, bc, out, m, n, k);
	// matmulCudaKernelGlobalT<T, mask_t><<<block_dims, thread_dims, 0, d.stream()>>>(ac, bc, out, m, n, k);
	// matmulCudaKernelGlobalT<T, unsigned long long><<<block_dims, thread_dims, 0, d.stream()>>>(ac64, bc64, out, m, n, k);
	// matmulCudaKernelShared<T><<<block_dims, thread_dims, 0, d.stream()>>>(ac, bc, out, m, n, k);
	matmulCudaKernelShared<T, mask_t><<<block_dims, thread_dims, 0, d.stream()>>>(ac, bc, out, m, n, k);
	
	// cudaFree(bc64);
	// cudaFree(ac64);
	// cudaFree(ac2);
	// cudaFree(bc);
	// cudaFree(ac);
}


// Explicitly instantiate functors for the types of OpKernels registered.
template struct XnorMatmulFunctor<GPUDevice, float, unsigned int>;
template struct XnorMatmulFunctor<GPUDevice, int32, unsigned int>;
template struct XnorMatmulFunctor<GPUDevice, float, unsigned long long>;
template struct XnorMatmulFunctor<GPUDevice, int32, unsigned long long>;

#endif  // GOOGLE_CUDA
