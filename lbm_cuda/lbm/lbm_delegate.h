#ifndef _LBM_DELEGATE_H_
#define _LBM_DELEGATE_H_

#include <cuda_runtime.h>
#include <GL/freeglut.h>

#define LATTICE_WIDTH 256
#define LATTICE_HEIGHT 128
//#define LATTICE_DEPTH 64
#define LATTICE_DIMENSION 32768
#define INDEX(x, y) ((x) + (y) * LATTICE_WIDTH)

typedef enum {
	CURL,
	SPEED,
	UX,
	UY
} render_mode;

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
