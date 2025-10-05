#ifndef _D2Q9_H_
#define _D2Q9_H_

#include "lbm.h"

#define D2Q9_INDEX(x, y) ((x) + (y) * LATTICE_WIDTH)

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

#endif
