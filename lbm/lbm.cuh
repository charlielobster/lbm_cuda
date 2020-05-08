#ifndef _LBM_CUH_
#define _LBM_CUH_

#define VISCOSITY 0.005
#define CONTRAST 75
#define VELOCITY 0.1
#define VELOCITY_SQUARED 0.01
#define STEPS_PER_RENDER 10
#define LATTICE_WIDTH 256
#define LATTICE_HEIGHT 128
//#define LATTICE_DEPTH 64
#define LATTICE_DIMENSION 32768
#define INDEX(x, y) ((x) + (y) * LATTICE_WIDTH)
#define CLIP(n) ((n) > 255 ? 255 : ((n) < 0 ? 0 : (n)))

enum direction {
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

enum renderMode {
	CURL,
	SPEED,
	UX,
	UY
};

typedef struct {
	//velocities:
	float ux;	//x velocity
	float uy;	//y velocity
	float rho;	//density. aka rho
	float f[9];
} lbm_node;

typedef struct {
	char ex; //x location
	char ey; //y location
	float wt; //weight
	unsigned char op; //opposite char
} d2q9_node;

#endif