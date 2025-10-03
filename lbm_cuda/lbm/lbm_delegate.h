#ifndef _LBM_DELEGATE_H_
#define _LBM_DELEGATE_H_

#include <cuda_runtime.h>
#include <GL/freeglut.h>

#include "lbm.h"

class lbm_delegate
{
public:
	lbm_delegate() {}
	virtual ~lbm_delegate() {}
	virtual void launchKernels(render_mode mode, bool barriersUpdated, unsigned char* barrier, unsigned char* out) = 0;
	virtual void resetLattice(GLuint pbo, unsigned char* barrier) = 0;
	virtual void freeCUDA() = 0;
};

#endif
