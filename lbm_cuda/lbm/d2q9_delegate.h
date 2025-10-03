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
} d2q9_position_weight;

// encapsulate the d2q9 details
class d2q9_delegate : public lbm_delegate
{
public:
	d2q9_delegate() : a1(0), a2(0), array1_gpu(0), array2_gpu(0), barrier_gpu(0), d2q9_gpu(0), cuda_pbo_resource(0) {}
	~d2q9_delegate() {}

	void launchKernels(render_mode mode, bool barriersUpdated, unsigned char* barrier, unsigned char* out);
	void resetLattice(GLuint pbo, unsigned char* barrier);
	void freeCUDA();

private:
	void initCUDA(d2q9_position_weight* d2q9, d2q9_lbm_node* array1, d2q9_lbm_node* array2, unsigned char* barrier);
	void initPboResource(GLuint pbo);
	void initD2q9(d2q9_position_weight* d2q9);
	void initA1(d2q9_position_weight* d2q9);

	d2q9_lbm_node* a1;
	d2q9_lbm_node* a2;

	d2q9_lbm_node* array1_gpu;
	d2q9_lbm_node* array2_gpu;
	unsigned char* barrier_gpu;
	d2q9_position_weight* d2q9_gpu;
	struct cudaGraphicsResource* cuda_pbo_resource;
};

#endif
