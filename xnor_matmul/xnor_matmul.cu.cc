#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "xnor_matmul.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#define BLOCK_SIZE 32


template <typename T>
__device__ unsigned int signArraytoBitmask(T* array, int len)
{
	unsigned int bitmask = 0;
    unsigned int sign;
    
    for (int i = 0; i < len; ++i)
    {
        sign = (array[i] >= 0);				
        bitmask = bitmask | (sign<<i);
    }
    
	return bitmask;
}


template <typename T>
__global__ void concantenateRowsSigns(T* mtx, unsigned int* sign_mtx, int size)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(tid < size)
	{
		const T* sub_mtx = mtx + tid * 32;
		sign_mtx[tid] = signArraytoBitmask(sub_mtx, 32);
	}
	
}


template <typename T>
__global__ void concantenateColumnsSigns(T* mtx, unsigned int* sign_mtx, int m, int n, int size)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// in order to avoid other memory alloctions, the sign bitmask generations
	// is done directly in this function (without calling the previous dedicated function)
	
	unsigned int bitmask = 0;
	unsigned int sign;
	
	if(tid < size)
	{
		for(int i=0; i<32; ++i)
		{
			int col = (tid*32 + i) / m;
			int row = (tid*32 + i) % m;
			sign = (mtx[row * n + col] >= 0);
			bitmask = bitmask | (sign<<i);
		}
		sign_mtx[tid] = bitmask;
	}
}



template <typename T>
__global__ void matmulCudaKernel(unsigned int* a_mtx, unsigned int* b_mtx, T* c_mtx, int m, int n, int k)
{
	
	int block_col = blockIdx.x;
	int block_row = blockIdx.y;
	
	int col = threadIdx.x;
	int row = threadIdx.y;
		
	// T* Csub = c_mtx + (block_row * blockDim.y * k + block_col * blockDim.x);
	
	// // each thread copy its submatrix data in in shared memory
	// // each A submatrix block is BLOCK_SIZE*32 --> hence BLOCK_SIZE wide unsigned int (32 bits) array
	// // each B submatrix block is 32*BLOCK_SIZE --> hence BLOCK_SIZE wide unsigned int (32 bits) array
	// // __shared__ unsigned int Asub[BLOCK_SIZE];
	// // __shared__ unsigned int Bsub[BLOCK_SIZE];
	
	// unsigned int c_value = 0;
	
	// // computing submatrix C_xy associated to the thread block as sum_i(A_xi * B_iy)
	// for(int i = 0; i < n / 32; ++i)
	// {
		// // getting firt element pointer of submatrix A_xi and B_iy
		// unsigned int* As = a_mtx + (block_row * blockDim.y * n / 32 + blockDim.x * i); 	// a_mtx is a concatenation of rows with 32 elements grouped in one uint
		// unsigned int* Bs = b_mtx + (i * BLOCK_SIZE * k + BLOCK_SIZE * block_col);		// b_mtx is a concatenation of columns with 32 elements grouped in one uint
		
		// // copy submatrix A_xi and submatrix B_iy to shared memory (each thread one element)
		// // Asub[row] = As[row * n + col];
		// // Bsub[col] = Bs[row * k + col];
		
		// // evaluating c_value only after all the block threads have copied their part of A and B in shared memory
		// // __syncthreads();
		
		// // c_value += __popc((Asub[row] ^ Bsub[col]));
		// c_value += __popc((As[row * n + col] ^ Bs[row * k + col]));
		
		// // __syncthreads();
	// }	
	
	// Csub[row * k + col] = c_value - (n - c_value);
	
	
	// NAIVE PARALLEL MULTIPLICATION CODE (NOT OPTIMIZED)
	int trow = block_row * blockDim.y + row;
	int tcol = block_col * blockDim.x + col;
	
	if(trow >= m || tcol >= k)
		return;
	
	unsigned int* Ar = a_mtx + (trow * (n / 32));
	unsigned int* Bc = b_mtx + (tcol * (n / 32));
	unsigned int c_value = 0;
	
	for(int i = 0; i < n / 32; ++i)
		c_value += __popc(Ar[i] ^ Bc[i]);
	
	c_mtx[trow * k + tcol] = -(static_cast<T>(2 * c_value) - n);
}



// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void XnorMatmulFunctor<GPUDevice, T>::operator()(const GPUDevice& d, T* a_mtx, T* b_mtx, T* out, int m, int n, int k)
{
	
	// // allocate memory for concatenated A matrix and b matrix
	unsigned int* ac;
	unsigned int* bc;
	
	cudaMalloc((void**)&ac, m*n*sizeof(unsigned int) / 32);
	cudaMalloc((void**)&bc, n*k*sizeof(unsigned int) / 32);
	
	int thread_per_block = 32;
	int block_count = (m * n) / (thread_per_block * 32) + 1;
	concantenateRowsSigns<T> <<<block_count, thread_per_block, 0, d.stream()>>> (a_mtx, ac, (m*n)/32);
	
	block_count = (n * k) / (thread_per_block * 32) + 1;
	concantenateColumnsSigns<T> <<<block_count, thread_per_block, 0, d.stream()>>> (b_mtx, bc, n, k, (n*k)/32);
	
	dim3 block_dims(k/32, m/BLOCK_SIZE);
	dim3 thread_dims(32, BLOCK_SIZE);
	matmulCudaKernel<T><<<block_dims, thread_dims, 0, d.stream()>>>(ac, bc, out, m, n, k);
	
	cudaFree(bc);
	cudaFree(ac);
}


// Explicitly instantiate functors for the types of OpKernels registered.
template struct XnorMatmulFunctor<GPUDevice, float>;
template struct XnorMatmulFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
