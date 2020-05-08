#ifndef _LBM_H_
#define _LBM_H_

#define VISCOSITY 0.005
#define CONTRAST 75
#define VELOCITY 0.1
#define VELOCITY_SQUARED 0.01
#define _3V 0.3
#define _3V2 0.03
#define STEPS_PER_RENDER 10
#define LATTICE_WIDTH 256
#define LATTICE_HEIGHT 128
//#define LATTICE_DEPTH 64
#define LATTICE_DIMENSION 32768
#define INDEX(x, y) ((x) + (y) * LATTICE_WIDTH)
#define CLIP(n) ((n) > 255 ? 255 : ((n) < 0 ? 0 : (n)))

enum heading {
	NONE = 0,
	EAST,
	NORTH,
	WEST,
	SOUTH,
	NORTHEAST,
	NORTHWEST,
	SOUTHWEST,
	SOUTHEAST
};

enum render_mode {
	CURL,
	SPEED,
	UX,
	UY
};

typedef struct {
	float ux;	// x velocity
	float uy;	// y velocity
	float rho;	// density
	float direction[9];
} lbm_node;

typedef struct {
	char x_position;			
	char y_position;
	float weight;
	unsigned char opposite;	// opposite char
} d2q9_node;

class lbm 
{
public:
	lbm() {}
	~lbm() {}
	static void printDeviceInfo();
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