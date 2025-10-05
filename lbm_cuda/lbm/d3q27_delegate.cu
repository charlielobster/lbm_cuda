#include <helper_gl.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "d3q27_delegate.h"
#include "d3q27_global.cuh"

void d3q27_delegate::launchKernels(lbm_render_mode mode)
{

}

void d3q27_delegate::resetLattice(GLuint pbo)
{

}

void d3q27_delegate::freeCUDA()
{
	cudaFree(d3q27_gpu);
	cudaFree(array1_gpu);
	cudaFree(array2_gpu);
	cudaFree(barrier_gpu);
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
}

