#ifndef _D3Q27_H_
#define _D3Q27_H_

#include "lbm.h"

#define D3Q27_INDEX(x, y, z) ((z) + (x) * LATTICE_HEIGHT + (y) * LATTICE_WIDTH * LATTICE_HEIGHT)

typedef enum {
	ZERO = 0,
	E, 		// 1
	W, 		// 2
	N, 		// 3
	S, 		// 4
	U, 		// 5
	L, 		// 6
	NE, 	// 7
	SW, 	// 8
	UE, 	// 9
	LW, 	// 10
	UN, 	// 11
	LS, 	// 12
	SE, 	// 13
	NW, 	// 14
	LE,	 	// 15
	UW, 	// 16
	LN, 	// 17
	US, 	// 18
	UNE, 	// 19
	LSW, 	// 20
	LNE, 	// 21
	USW, 	// 22
	USE, 	// 23
	LNW, 	// 24
	UNW, 	// 25
	LSE 	// 26
} d3q27_vector;

typedef struct {
	float ux;	// x velocity
	float uy;	// y velocity
	float uz;	// z velocity
	float rho;	// density
	float vectors[27];
} d3q27_lbm_node;

typedef struct {
	char x_position;			
	char y_position;
	char z_position;
	float weight;
	unsigned char opposite;
} d3q27_velocity_set;

#endif
