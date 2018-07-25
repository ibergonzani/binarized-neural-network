#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "xnor_gemm.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#define BLOCK_SIZE 16


// 32 single float array ->  32 bits unsigned int
template <typename T>
__device__ unsigned int concatenate(T* array)
{
    unsigned int rvalue=0;
    unsigned int sign;
    
    for (int i = 0; i < 32; i++)
    {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<<i);
    }
    
    return rvalue;
}

template <typename T>
__global__ void concatenate_rows_kernel(T *a, unsigned int *b, int size)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size) b[i] = concatenate(&a[i*32]);
}

template <typename T>
__global__ void concatenate_cols_kernel(T *a, unsigned int *b, int m, int n)
{   

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(j<n){
        T * array = new T[32];
        for(int i=0; i<m; i+=32){
            for(int k=0; k<32;k++) array[k] = a[j + n*(i+k)];
            b[j+n*i/32]=concatenate(array); 
        } 
        delete[] array;
    }
}

// 32 bits unsigned int -> 32 single float array
// TODO: the array allocation should not be done here
template <typename T>
__device__ T* deconcatenate(unsigned int x)
{
    T* array = new float[32];
    
    for (int i = 0; i < 32; i++)    
    {   
        array[i] = (x & ( 1 << i )) >> i;
    }
    
    return array;
}

template <typename T>
__global__ void deconcatenate_rows_kernel(unsigned int* a, T* b, int size)
{ 
    T* array;
    
    for(int i=0; i<size; i+=32)
    {
        array = deconcatenate<T>(a[i/32]);
        for (int k=0;k<32;k++) b[i+k] = array[k];
        delete[] array;
    }
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
    T* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

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
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = -(2*(T)Cvalue-32*n);
}



// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void XnorGemmFunctor<GPUDevice, T>::operator()(const GPUDevice& d, T* a_mtx, T* b_mtx, T* out, int m, int n, int k)
{
	
	// allocate memory for concatenated A matrix and b matrix
	unsigned int* ac;
	unsigned int* bc;
	
	cudaMalloc((void**)&ac, m*n*sizeof(unsigned int) / 32);
	cudaMalloc((void**)&bc, n*k*sizeof(unsigned int) / 32);
	
	
	int block_size = 64;
	dim3 block = dim3(block_size,1,1);
	dim3 grid = dim3(m*n/(block_size*32)+1,1);
	concatenate_rows_kernel<T> <<<grid, block, 0, d.stream()>>> (a_mtx, ac, m*n/32);
	
	block_size = 64;
	block = dim3(block_size,1,1);
	grid = dim3(k/block_size+1,1);
	concatenate_cols_kernel<T> <<<grid, block, 0, d.stream()>>> (b_mtx, bc, m, n);
	
	block_size = 16;
	block = dim3(block_size, block_size, 1);
	grid = dim3(k/block_size + 1, m/block_size + 1);
	xnor_gemm<T><<<grid, block, 0, d.stream()>>>(ac, bc, out, m, n, k);
	
	cudaFree(bc);
	cudaFree(ac);
}


// Explicitly instantiate functors for the types of OpKernels registered.
template struct XnorGemmFunctor<GPUDevice, float>;
template struct XnorGemmFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
