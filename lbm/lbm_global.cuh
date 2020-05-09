#ifndef _LBM_GLOBAL_CUH_
#define _LBM_GLOBAL_CUH_

#include "lbm_device.cuh"

__global__
static void collide(d2q9_node* d2q9, lbm_node* before, lbm_node* after, unsigned char* barrier)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = INDEX(x, y);

	float omega = 1 / (3 * VISCOSITY + 0.5);

	//toss out out of bounds
	if (x < 0 || x >= LATTICE_WIDTH || y < 0 || y >= LATTICE_HEIGHT)
		return;

	lbm_device::macroGen(before[i].direction, &(after[i].ux), &(after[i].uy), &(after[i].rho), i);

	int dir = 0;
	for (dir = 0; dir < 9; dir += 1)
	{
		after[i].direction[dir] = before[i].direction[dir] + omega
			* (lbm_device::accelGen(dir, after[i].ux, after[i].uy,
				after[i].ux * after[i].ux + after[i].uy
				* after[i].uy, after[i].rho, d2q9) - before[i].direction[dir]);
	}
	return;
}

//stream: handle particle propagation, ignoring edge cases.
__global__
static void stream(d2q9_node* d2q9, lbm_node* before, lbm_node* after, unsigned char* barrier)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = INDEX(x, y);

	//toss out out of bounds and edge cases
	if (x < 0 || x >= LATTICE_WIDTH || y < 0 || y >= LATTICE_HEIGHT)
		return;

	after[i].rho = before[i].rho;
	after[i].ux = before[i].ux;
	after[i].uy = before[i].uy;

	if (!(x > 0 && x < LATTICE_WIDTH - 1 && y > 0 && y < LATTICE_HEIGHT - 1))
	{
		//return;
		lbm_device::streamEdgeCases(x, y, after, barrier, d2q9);
	}
	else
	{
		//propagate all f values around a bit
		int dir = 0;
		for (dir = 0; dir < 9; dir += 1)
		{
			after[INDEX(d2q9[dir].x_position + x, -d2q9[dir].y_position + y)].direction[dir] =
				before[i].direction[dir];
		}
	}
}

__global__
static void bounce(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
		unsigned char* barrier, uchar4* image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = INDEX(x, y);

	if (x > 0 && x < LATTICE_WIDTH - 1 && y > 0 && y < LATTICE_HEIGHT - 1)
	{
		if (barrier[i] == 1)
		{
			int d;
			for (d = 1; d < 9; d += 1)
			{
				if (d2q9[d].opposite > 0 && (after[i].direction)[d] > 0)
				{
					after[INDEX(d2q9[d].x_position + x, -d2q9[d].y_position + y)].direction[d]
						= (before[i].direction)[d2q9[d].opposite];
				}
			}
		}
	}
}

__global__
static void color(render_mode mode, lbm_node* array, uchar4* image, unsigned char* barrier)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// toss out out of bounds and edge cases
	if (x < 0 || x >= LATTICE_WIDTH || y < 0 || y >= LATTICE_HEIGHT)
		return;

	int i = INDEX(x, y);

	if (barrier[i] == 1)
	{
		image[i].w = 255;
		image[i].x = 255;
		image[i].y = 255;
		image[i].z = 255;
	}
	else
	{
		switch (mode)
		{
		case CURL:
			image[i] = lbm_device::getRgbCurl(x, y, array);
			break;
		case SPEED:
			image[i] = lbm_device::getRgbU(sqrt(array[i].ux * array[i].ux + array[i].uy * array[i].uy));
			break;
		case UX:
			image[i] = lbm_device::getRgbU(array[i].ux);
			break;
		case UY:
			image[i] = lbm_device::getRgbU(array[i].uy);
			break;
		default:
			break;
		}
	}
}

#endif