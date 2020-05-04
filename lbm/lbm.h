#ifndef _LBM_H_
#define _LBM_H_

#define trace_x 50
#define trace_y 57
#define DEBUG_DELAY 0

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

enum directions {
	d0 = 0,
	E,
	N,
	W,
	S,
	NE,
	NW,
	SW,
	SE
};

enum render_modes {
	Rho,
	Curl,
	Speed,
	Ux,
	Uy
};

typedef struct {
	float ux;	// x velocity
	float uy;	// y velocity
	float rho;	// density. aka rho
	float f[9];
} lbm_node;

typedef struct {
	char ex;	// x location
	char ey;	// y location
	float wt;	// weight
	unsigned char op; // opposite char
} d2q9_node;

typedef struct {
	float viscosity;
	float omega;
	unsigned int height;
	unsigned int width;
	float contrast;
	float v;
	unsigned char mode;
	unsigned int stepsPerRender;
} parameter_set;

#endif
