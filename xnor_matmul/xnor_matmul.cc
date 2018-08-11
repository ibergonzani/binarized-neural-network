#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "xnor_matmul.h"


using namespace tensorflow;


REGISTER_OP("XnorMatmul")
	.Attr("T: {float, int32}")
	.Attr("group64: bool = false")
	.Input("a: T")
	.Input("b: T")
	.Output("c: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		
		shape_inference::ShapeHandle a_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a_shape));
	 
		shape_inference::ShapeHandle b_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b_shape));
		
		shape_inference::DimensionHandle output_rows = c->Dim(a_shape, 0);
		shape_inference::DimensionHandle output_cols = c->Dim(b_shape, 1);
	 
		c->set_output(0, c->Matrix(output_rows, output_cols));
		return Status::OK();
		
});


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T, typename Mask>
struct XnorMatmulFunctor<CPUDevice, T, Mask> {
	void operator()(const CPUDevice& d, T* a_mtx, T* b_mtx, T* out, Mask* a_msk, Mask* b_msk, int m, int n, int k) {

	}
};

	
	
template <typename Device, typename T>
class XnorMatmul : public OpKernel {
	
	private:
		bool group64;
	
	public:
	
	explicit XnorMatmul(OpKernelConstruction* context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("group64", &group64));
	}
	
	// Operation inmplementation. Calls the correct template method based on the device
	void Compute(OpKernelContext* context) override {
		
		using mask_t = unsigned long long; 
		
		// matrix tensors to be multiplicated together
		const Tensor& a_mtx_tensor = context->input(0);
		const Tensor& b_mtx_tensor = context->input(1);
		
		// getting matrices dimesions
		const TensorShape& a_shape = a_mtx_tensor.shape();
		const TensorShape& b_shape = b_mtx_tensor.shape();
		
		int m = a_shape.dim_size(0);	int j = b_shape.dim_size(0);
		int n = a_shape.dim_size(1);	int k = b_shape.dim_size(1);
		
		//checking matrix dimensions
		DCHECK_EQ(n, j);
		// DCHECK_EQ(m % 16, 0);
		// DCHECK_EQ(k % 16, 0);
		
		// Creating and allocating the output tensor
		TensorShape output_shape;
		output_shape.AddDim(m);
		output_shape.AddDim(k);
		
		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
		
		// Creating and allocating support tensors		
		Tensor* a_cct_tensor = NULL;
		PersistentTensor persistent_a_cct;
		TensorShape ac_mask_shape({m * n / 64});
		OP_REQUIRES_OK(context, context->allocate_persistent(DT_UINT64, ac_mask_shape, &persistent_a_cct, &a_cct_tensor));
		
		Tensor* b_cct_tensor = NULL;
		PersistentTensor persistent_b_cct;
		TensorShape bc_mask_shape({j * k / 64});
		OP_REQUIRES_OK(context, context->allocate_persistent(DT_UINT64, bc_mask_shape, &persistent_b_cct, &b_cct_tensor));
		
		
		// calling multiplication kernel (on CPU or GPU)
		if(group64)
		{
			using mask_t = unsigned long long;
			
			DCHECK_EQ(n % (sizeof(mask_t)*8), 0);
			
			XnorMatmulFunctor<Device, T, mask_t>()(
				context->eigen_device<Device>(),
				(T*) &(a_mtx_tensor.flat<T>()(0)),
				(T*) &(b_mtx_tensor.flat<T>()(0)),
				(T*) &(output_tensor->flat<T>()(0)),
				(mask_t*) &(a_cct_tensor->flat<unsigned long long>()(0)),
				(mask_t*) &(b_cct_tensor->flat<unsigned long long>()(0)),
				m,
				n,
				k);
		}
		else
		{
			using mask_t = unsigned int;
						
			DCHECK_EQ(n % (sizeof(mask_t)*8), 0);
			
			XnorMatmulFunctor<Device, T, mask_t>()(
				context->eigen_device<Device>(),
				(T*) &(a_mtx_tensor.flat<T>()(0)),
				(T*) &(b_mtx_tensor.flat<T>()(0)),
				(T*) &(output_tensor->flat<T>()(0)),
				(mask_t*) &(a_cct_tensor->flat<unsigned long long>()(0)),
				(mask_t*) &(b_cct_tensor->flat<unsigned long long>()(0)),
				m,
				n,
				k);
		}
	}
};



// Register the CPU kernels.
#define REGISTER_CPU(type)                                          	\
  REGISTER_KERNEL_BUILDER(                                       		\
      Name("XnorMatmul").Device(DEVICE_CPU).TypeConstraint<type>("T"), 	\
      XnorMatmul<CPUDevice, type>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA


#define REGISTER_GPU(type)     											 \
  extern template struct XnorMatmulFunctor<GPUDevice, type, unsigned int>;		 \
  extern template struct XnorMatmulFunctor<GPUDevice, type, unsigned long long>; \
  REGISTER_KERNEL_BUILDER(                                       		\
      Name("XnorMatmul").Device(DEVICE_GPU).TypeConstraint<type>("T"), 	\
      XnorMatmul<GPUDevice, type>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
