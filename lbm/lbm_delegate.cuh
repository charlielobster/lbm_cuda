#ifndef _LBM__DELEGATE_CUH_
#define _LBM_DELEGATE_CUH_

#include <GL/gl.h>
#include <cuda_runtime.h>
#include <helper_gl.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "lbm.h"

class lbm_delegate
{
public:
	//display stats of all detected cuda capable devices
	static void printDeviceInfo();

	lbm_delegate() : array1_gpu(0), array2_gpu(0), barrier_gpu(0), d2q9_gpu(0), cuda_pbo_resource(0) {}
	~lbm_delegate() {}

	void initPboResource(GLuint pbo);
	void initCUDA(d2q9_node* d2q9, lbm_node* array1, lbm_node* array2, unsigned char* barrier);
	void launchKernels(render_mode mode, bool barriersUpdated, unsigned char* barrier);
	void freeCUDA();

private:
	lbm_node* array1_gpu;
	lbm_node* array2_gpu;
	unsigned char* barrier_gpu;
	d2q9_node* d2q9_gpu;
	struct cudaGraphicsResource* cuda_pbo_resource;
};

#endif