#ifndef _LBM_H_
#define _LBM_H_

#include <cuda_runtime.h>

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

typedef enum {
	NONE = 0,
	EAST,
	NORTH,
	WEST,
	SOUTH,
	NORTHEAST,
	NORTHWEST,
	SOUTHWEST,
	SOUTHEAST
} heading;

typedef enum {
	CURL,
	SPEED,
	UX,
	UY
} render_mode;

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
	unsigned char opposite;
} d2q9_node;

#endif