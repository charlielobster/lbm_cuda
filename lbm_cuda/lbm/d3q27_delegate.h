#ifndef _D2Q27_DELEGATE_H_
#define _D2Q27_DELEGATE_H_

#include <cuda_runtime.h>
#include <GL/freeglut.h>

#include "lbm_delegate.h"
#include "d3q27.h"

// encapsulate the d3q27 details
class d3q27_delegate : public lbm_delegate
{
public:
	d3q27_delegate() :
		array1(0), 
		array2(0), 
		barrier(0), 
		array1_gpu(0), 
		array2_gpu(0), 
		barrier_gpu(0), 
		d3q27_gpu(0), 
		cuda_pbo_resource(0), 
		barrierUpdated(true) {}
	~d3q27_delegate() {}
	void launchKernels(lbm_render_mode mode);
	void resetLattice(GLuint pbo);
	void freeCUDA();

private:
	bool barrierUpdated;
	unsigned char* barrier;	
	d3q27_lbm_node* array1;
	d3q27_lbm_node* array2;
	d3q27_lbm_node* array1_gpu;
	d3q27_lbm_node* array2_gpu;
	unsigned char* barrier_gpu;
	d3q27_velocity_set* d3q27_gpu;
	struct cudaGraphicsResource* cuda_pbo_resource;
};

#endif
