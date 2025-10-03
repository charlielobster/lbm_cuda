#ifndef _LBM_DEVICE_CUH_
#define _LBM_DEVICE_CUH_

#include "lbm.h"

class lbm_device
{
public:
	__device__
	static uchar4 getRgbU(float i)
	{

		uchar4 val;
		if (i == i)
		{
			val.w = 255;
			val.x = 0;
			val.y = CLIP(i * 255.0 / 1.0);
			val.z = 0;
		}
		else
		{
			val.w = 255;
			val.x = 255;
			val.y = 0;
			val.z = 0;
		}
		return val;
	}

	__device__
	static float computeCurlMiddleCase(int x, int y, d2q9_lbm_node* array1)
	{
		return (array1[INDEX(x, y + 1)].ux - array1[INDEX(x, y - 1)].ux)
			- (array1[INDEX(x + 1, y)].uy - array1[INDEX(x - 1, y)].uy);
	}

	__device__
	static uchar4 getRgbCurl(int x, int y, d2q9_lbm_node* array)
	{
		int i = INDEX(x, y);
		uchar4 val;
		val.x = 0;
		val.w = 255;
		if (0 < x && x < LATTICE_WIDTH - 1)
		{
			if (0 < y && y < LATTICE_HEIGHT - 1)
			{
				if (computeCurlMiddleCase(x, y, array) > 0)
				{
					val.y = CLIP(20 * CONTRAST * computeCurlMiddleCase(x, y, array));
					val.z = 0;
				}
				else
				{
					val.z = CLIP(20 * CONTRAST * -1 * computeCurlMiddleCase(x, y, array));
					val.y = 0;
				}
			}
		}

		if (array[i].rho != array[i].rho)
		{
			val.x = 255;
			val.y = 0;
			val.z = 0;
			val.w = 255;
		}
		return val;
	}

	__device__
	static void macroGen(float* f, float* ux, float* uy, float* rho, int i)
	{
		const float top_row = f[6] + f[2] + f[5];
		const float mid_row = f[3] + f[0] + f[1];
		const float bot_row = f[7] + f[4] + f[8];

		*rho = top_row + mid_row + bot_row;
		if (*rho > 0)
		{
			*ux = ((f[5] + f[1] + f[8]) - (f[6] + f[3] + f[7])) / (*rho);
			*uy = (bot_row - top_row) / (*rho);
		}
		else
		{
			*ux = 0;
			*uy = 0;
		}

		return;
	}

	// return acceleration
	__device__
	static float accelGen(int node_num, float ux, float uy, float u2, float rho, d2q9_position_weight* d2q9)
	{
		float u_direct = ux * d2q9[node_num].x_position + uy * (-d2q9[node_num].y_position);
		float unweighted = 1 + 3 * u_direct + 4.5 * u_direct * u_direct - 1.5 * u2;

		return rho * d2q9[node_num].weight * unweighted;
	}

	__device__
	static void doLeftWall(int i, d2q9_lbm_node* after, d2q9_position_weight* d2q9)
	{
		after[i].vectors[EAST] = d2q9[EAST].weight * (1 + _3V + _3V2);
		after[i].vectors[NORTHEAST] = d2q9[NORTHEAST].weight * (1 + _3V + _3V2);
		after[i].vectors[SOUTHEAST] = d2q9[SOUTHEAST].weight * (1 + _3V + _3V2);
	}

	__device__
	static void doRightWall(int i, d2q9_lbm_node* after, d2q9_position_weight* d2q9)
	{
		after[i].vectors[WEST] = d2q9[WEST].weight * (1 - _3V + _3V2);
		after[i].vectors[NORTHWEST] = d2q9[NORTHWEST].weight * (1 - _3V + _3V2);
		after[i].vectors[SOUTHWEST] = d2q9[SOUTHWEST].weight * (1 - _3V + _3V2);
	}

	// top and bottom walls
	__device__
	static void doFlanks(int i, d2q9_lbm_node* after, d2q9_position_weight* d2q9)
	{
		after[i].vectors[ZERO] = d2q9[ZERO].weight * (1 - 1.5 * VELOCITY_SQUARED);
		after[i].vectors[EAST] = d2q9[EAST].weight * (1 + _3V + _3V2);
		after[i].vectors[WEST] = d2q9[WEST].weight * (1 - _3V + _3V2);
		after[i].vectors[NORTH] = d2q9[NORTH].weight * (1 - 1.5 * VELOCITY_SQUARED);
		after[i].vectors[SOUTH] = d2q9[SOUTH].weight * (1 - 1.5 * VELOCITY_SQUARED);
		after[i].vectors[NORTHEAST] = d2q9[NORTHEAST].weight * (1 + _3V + _3V2);
		after[i].vectors[SOUTHEAST] = d2q9[SOUTHEAST].weight * (1 + _3V + _3V2);
		after[i].vectors[NORTHWEST] = d2q9[NORTHWEST].weight * (1 - _3V + _3V2);
		after[i].vectors[SOUTHWEST] = d2q9[SOUTHWEST].weight * (1 - _3V + _3V2);
	}

	__device__
	static void streamEdgeCases(int x, int y, d2q9_lbm_node* after, unsigned char* barrier, d2q9_position_weight* d2q9)
	{
		int i = INDEX(x, y);
		if (x == 0)
		{
			if (barrier[i] != 1)
			{
				doLeftWall(i, after, d2q9);
			}
		}
		else if (x == LATTICE_WIDTH - 1)
		{
			if (barrier[i] != 1)
			{
				doRightWall(i, after, d2q9);
			}
		}
		else if (y == 0 || y == LATTICE_WIDTH - 1)
		{
			if (barrier[i] != 1)
			{
				doFlanks(i, after, d2q9);
			}
		}
	}
};

#endif