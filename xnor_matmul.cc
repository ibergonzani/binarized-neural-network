#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

REGISTER_OP("XNORmatmul")
	.Input("A: int32")
	.Input("B: int32")
	.Output("product: int32")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(0));
		return Status::OK();
});


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct XNORmatmulFunctor<CPUDevice, T> {
	void operator()(const CPUDevice& d, int size, const T* in, T* out) {
		for (int i = 0; i < size; ++i) {
			out[i] = 2 * in[i];
		}
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

		// Creating and allocating the output tensor
		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

		OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
					errors::InvalidArgument("Too many elements in tensor"));
		
		XNORmatmulFunctor<Device, T>()(
			context->eigen_device<Device>(),
			static_cast<int>(input_tensor.NumElements()),
			a_mtx_tensor.flat<T>().data(),
			b_mtx_tensor.flat<T>().data(),
			output_tensor->flat<T>().data());
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
