#ifndef _LBM_DELEGATE_H_
#define _LBM_DELEGATE_H_

#include <GL/freeglut.h>

typedef enum {
	CURL,
	SPEED,
	UX,
	UY
} lbm_render_mode;

class lbm_delegate
{
public:
	lbm_delegate() {}
	virtual ~lbm_delegate() {}
	virtual void launchKernels(lbm_render_mode mode) = 0;
	virtual void resetLattice(GLuint pbo) = 0;
	virtual void clearBarrier() = 0;
	virtual void drawLineDiagonal() = 0;
	virtual void drawSquare() = 0;
	virtual void freeCUDA() = 0;
};

#endif
