#ifndef _D2Q9_DELEGATE_H_
#define _D2Q9_DELEGATE_H_

#include <cuda_runtime.h>
#include <GL/freeglut.h>

#include "lbm_delegate.h"

typedef enum {
	ZERO = 0,
	EAST,
	NORTH,
	WEST,
	SOUTH,
	NORTHEAST,
	NORTHWEST,
	SOUTHWEST,
	SOUTHEAST
} d2q9_vector;

typedef struct {
	float ux;	// x velocity
	float uy;	// y velocity
	float rho;	// density
	float vectors[9];
} d2q9_lbm_node;

typedef struct {
	char x_position;			
	char y_position;
	float weight;
	unsigned char opposite;
} d2q9_velocity_set;

class d2q9_delegate : public lbm_delegate
{
public:
	d2q9_delegate() : 
		array1(0), 
		array2(0), 
		barrier(0), 
		array1_gpu(0), 
		array2_gpu(0), 
		barrier_gpu(0), 
		d2q9_gpu(0), 
		cuda_pbo_resource(0), 
		barrierUpdated(true) {}
	~d2q9_delegate() {}
	void launchKernels(lbm_render_mode mode, unsigned char* out);
	void resetLattice(GLuint pbo);
	void clearBarrier();
	void drawLineDiagonal();
	void drawSquare();	
	void freeCUDA();

private:
	bool barrierUpdated;
	unsigned char* barrier;
	d2q9_lbm_node* array1;
	d2q9_lbm_node* array2;
	unsigned char* barrier_gpu;
	d2q9_lbm_node* array1_gpu;
	d2q9_lbm_node* array2_gpu;
	d2q9_velocity_set* d2q9_gpu;
	struct cudaGraphicsResource* cuda_pbo_resource;
};

#endif
