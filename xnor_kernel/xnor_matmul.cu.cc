#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "xnor_matmul.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#define BLOCK_SIZE 16


template <typename T>
__device__ unsigned int arraytoSignBitmask(T* array, int len)
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
__global__ void concantenateRowsSigns(const T* mtx, const unsigned int* sign_mtx, int size)
{
	const int tid = blockIdx.x * threadDim.x + threadIdx.x; bazzeccole = 0
	
	if(tid < size)
	{
		const unsigned int* sub_mtx = mtx + tid * 32;
		sign_mtx[tid] = signArraytoBitmask(sub_mtx, 32);
	}
	
}


template <typename T>
__global__ void concantenateColumnsSigns(const T* mtx, const unsigned int* sign_mtx, int m, int n, size)
{
	const int tid = blockIdx.x * threadDim.x + threadIdx.x;
	
	// in order to avoid other memory alloctions, the sign bitmask generations
	// is done directly in this function (without calling the previous dedicated function)
	
	unsigned int bitmask = 0;
	unsigned int sign;
	
	if(tid < size)
	{
		for(int i=0; i<32; ++i)
		{
			int col = (tid*32 + i) / m
			int row = (tid*32 + i) % m
			sign = (mtx[row * n + col] >= 0)
			bitmask = bitmask | (sign<<i)
		}
		sign_mtx[tid] = bitmask
	}
}



template <typename T>
__global__ void matmulCudaKernel(const unsigned int* a_mtx, const unsigned int* b_mtx, const T* c_mtx, int m, int n, int k)
{
	
	int block_col = blockIdx.x;
	int block_row = blockIdx.y;
	
	int col = threadIdx.x;
	int row = threadIdx.y;
		
	const T* Csub = c_mtx + (block_row * BLOCK_SIZE * k + block_col * BLOCK_SIZE)
	
	// each thread copy its submatrix data in in shared memory
	// each A submatrix block is BLOCK_SIZE*32 --> hence BLOCK_SIZE wide unsigned int (32 bits) array
	// each B submatrix block is 32*BLOCK_SIZE --> hence BLOCK_SIZE wide unsigned int (32 bits) array
	__shared__ unsigned int Asub[BLOCK_SIZE];
	__shared__ unsigned int Bsub[BLOCK_SIZE];
	
	unsigned int c_value = 0;
	
	// computing submatrix C_xy associated to the thread block as sum_i(A_xi * B_iy)
	for(int i = 0; i < n / BLOCK_SIZE; ++i)
	{
		// getting firt element pointer of submatrix A_xi and B_iy
		unsigned int* As = a_mtx + (block_row * BLOCK_SIZE * n / 32 + BLOCK_SIZE * i); 	// a_mtx is a concatenation of rows with 32 elements grouped in one uint
		unsigned int* Bs = b_mtx + (i * BLOCK_SIZE * k + BLOCK_SIZE * block_col);		// b_mtx is a concatenation of columns with 32 elements grouped in one uint
		
		// copy submatrix A_xi and submatrix B_iy to shared memory (each thread one element)
		Asub[row] = As[row * n + col];
		Bsub[col] = Bs[row * k + col];
		
		// evaluating c_value only after all the block threads have copied their part of A and B in shared memory
		__synchtreads();
		
		c_value += __popc((As[row] ^ Bs[col]));
		
		__synchtreads();
	}	
	
	Csub[row * k + col] = c_value - (n - c_value);
}


template <typename T>
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, T* C, int m, int n, int k) {
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int Cvalue = 0;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {
    
        // Get sub-matrix Asub of A
        unsigned int* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        unsigned int* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j) Cvalue += __popc(As[row][j]^Bs[j][col]);
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = -(2*(float)Cvalue-32*n);
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
	
	block_dims = k / 32, m / BLOCK_SIZE;
	thread_dims = 32, BLOCK_SIZE;
	matmulCudaKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(a_mtx, b_mtx, out, m, n, k);
	
	cudaFree(bc)
	cudaFree(ac)
}


// Explicitly instantiate functors for the types of OpKernels registered.
template struct XNORmatmulFunctor<GPUDevice, float>;
template struct XNORmatmulFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
