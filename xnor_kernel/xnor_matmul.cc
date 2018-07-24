#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "xnor_matmul.h"


using namespace tensorflow;


REGISTER_OP("XNORmatmul")
	.Attr("T: {float, int32}")
	.Input("A: T")
	.Input("B: T")
	.Output("C: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(0));
		return Status::OK();
		
		shape_inference::ShapeHandle a_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a_shape));
	 
		shape_inference::ShapeHandle b_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b_shape));
		
		shape_inference::DimensionHandle output_rows = c->Dim(a_shape, 1);
		shape_inference::DimensionHandle output_cols = c->Dim(b_shape, 0);
		
		//shape_inference::DimensionHandle merged;
		//TF_RETURN_IF_ERROR(c->Merge(input_rows, weight_cols, &merged));
	 
		c->set_output(0, c->Matrix(output_rows, output_cols));
		return Status::OK();
		
});


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct XNORmatmulFunctor<CPUDevice, T> {
	void operator()(const CPUDevice& d, const T* a_mtx, const T* b_mtx, T* out, int m, int n, int k) {
		// for (int i = 0; i < size; ++i) {
			// out[i] = 2 * in[i];
		// }
	}
};

	
	
template <typename Device, typename T>
class XNORmatmul : public OpKernel {
	
	public:
	
	explicit XNORmatmul(OpKernelConstruction* context) : OpKernel(context) {}
	
	// Operation inmplementation. Calls the correct template method based on the device
	void Compute(OpKernelContext* context) override {
		
		// matrix tensors to be multiplicated together
		const Tensor& a_mtx_tensor = context->input(0);
		const Tensor& b_mtx_tensor = context->input(0);
		
		// getting matrices dimesions
		const TensorShape& a_shape = a_mtx_tensor.shape();
		const TensorShape& b_shape = b_mtx_tensor.shape();
		
		//checking matrix dimensions
		DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(0));
		DCHECK_EQ(a_shape.dim_size(1) % 32, 0);
		DCHECK_EQ(a_shape.dim_size(0) % 16, 0);
		DCHECK_EQ(b_shape.dim_size(1) % 16, 0);
		
		// Creating and allocating the output tensor
		TensorShape output_shape;
		output_shape.AddDim(a_shape.dim_size(0));
		output_shape.AddDim(b_shape.dim_size(1));
		
		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
		
		// calling multiplication kernel (on CPU or GPU)
		XNORmatmulFunctor<Device, T>()(
			context->eigen_device<Device>(),
			a_mtx_tensor.flat<T>().data(),
			b_mtx_tensor.flat<T>().data(),
			output_tensor->flat<T>().data(),
			a_shape.dim_size(0),
			a_shape.dim_size(1),
			b_shape.dim_size(0));
	}
};



// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("XNORmatmul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      XNORmatmul<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template XNORmatmulFunctor<GPUDevice, T>;               \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("XNORmatmul").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      XNORmatmul<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
