/* ============================================================ */
/* LATTICE BOLTZMANN SIMULATOR                                  */
/* GPU accelerated with CUDA                                    */
/*                                                              */
/* Copyright (c) 2017 Tom Scherlis and Henry Friedlander        */
/* For SSA Physics 3                                            */
/* ============================================================ */

//comment out this line to hide prints:
//#define DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

//Cuda includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "lbm.cuh"

extern parameter_set params;
extern lbm_node* array1;
extern lbm_node* array2;
extern lbm_node* array1_gpu;
extern lbm_node* array2_gpu;
extern unsigned char* barrier;
extern unsigned char* barrier_gpu;
extern d2q9_node* d2q9_gpu;
extern parameter_set* params_gpu;
extern char needsUpdate;
extern int prex;
extern int prey;
extern cudaError_t ierrAsync;
extern cudaError_t ierrSync;

__device__
unsigned char clip(int n) 
{
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}

//get 1d flat index from row and col
__device__
int getIndex(int x, int y, parameter_set* params)
{
	return y * params->width + x;
}

__device__
uchar4 getRGB_roh(float i, parameter_set* params)
{

	uchar4 val;
	if (i == i)
	{
		int j = (1 - i) * 255 * 10; // approximately -255 to 255;

		val.x = 0;
		val.w = 0;
		val.z = 255;

		if (j > 0)
		{
			val.y = clip(j);
			val.z = 0;
		}
		else
		{
			val.z = clip(-j);
			val.y = 0;
		}
	}
	else
	{
		val.y = 0;
		val.x = 255;
		val.w = 0;
		val.z = 255;
	}
	return val;
}

__device__
uchar4 getRGB_u(float i)
{

	uchar4 val;
	if (i == i)
	{
		val.w = 255;
		val.x = 0;
		val.y = clip(i*255.0 / 1.0);
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
float computeCurlMiddleCase(int x, int y, lbm_node * array1, parameter_set* params) 
{
	return (array1[getIndex(x, y + 1, params)].ux
		- array1[getIndex(x, y - 1, params)].ux)
		- (array1[getIndex(x + 1, y, params)].uy
			- array1[getIndex(x - 1, y, params)].uy);
}

__device__
uchar4 getRGB_curl(int x, int y, lbm_node* array, parameter_set* params)
{

	uchar4 val;
	val.x = 0;
	val.w = 255;
	if (0 < x && x < params->width - 1) {
		if (0 < y && y < params->height - 1) {
			if (computeCurlMiddleCase(x, y, array, params) > 0)
			{
				val.y = clip(20 * params->contrast * computeCurlMiddleCase(x, y, array, params));
				val.z = 0;
			}
			else
			{
				val.z = clip(20 * params->contrast * -1 * computeCurlMiddleCase(x, y, array, params));
				val.y = 0;
			}
		}
	}

	if (array[getIndex(x, y, params)].rho != array[getIndex(x, y, params)].rho)
	{
		val.x = 255;
		val.y = 0;
		val.z = 0;
		val.w = 255;
	}
	return val;
}

__device__
void computeColor(lbm_node* array, int x, int y, parameter_set* params, uchar4* image, unsigned char* barrier, int prex, int prey)
{
	int i = getIndex(x, y, params);
	int prei = getIndex(prex, prey, params);

	if (barrier[i] == 1)
	{
		image[i].w = 255;
		image[i].x = 255;
		image[i].y = 255;
		image[i].z = 255;
	}
	else
	{
		switch (params->mode)
		{
		case(mRho):
			image[i] = getRGB_roh(array[i].rho, params);
			break;
		case(mCurl):
			image[i] = getRGB_curl(x, y, array, params);
			break;
		case(mSpeed):
			image[i] = getRGB_u(sqrt(array[i].ux * array[i].ux + array[i].uy * array[i].uy));
			break;
		case(mUx):
			image[i] = getRGB_u(array[i].ux);
			break;
		case(mUy):
			image[i] = getRGB_u(array[i].uy);
			break;
		}
	}
	if (i == prei)
	{
		image[i].x = 255;
		image[i].y = 0;
		image[i].z = 0;
		image[i].w = 255;
	}
}

__device__
void macro_gen(float* f, float* ux, float* uy, float* rho, int i, parameter_set* params)
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

//return acceleration
__device__
float accel_gen(int node_num, float ux, float uy, float u2, float rho, d2q9_node* d2q9)
{
	float u_direct = ux * d2q9[node_num].ex + uy * (-d2q9[node_num].ey);
	float unweighted = 1 + 3 * u_direct + 4.5*u_direct*u_direct - 1.5*u2;

	return rho * d2q9[node_num].wt * unweighted;
}

__global__
void collide(d2q9_node* d2q9, lbm_node* before, lbm_node* after, parameter_set* params, unsigned char* barrier)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = getIndex(x, y, params);

	float omega = 1 / (3 * params->viscosity + 0.5);

	//toss out out of bounds
	if (x<0 || x >= params->width || y<0 || y >= params->height)
		return;

	macro_gen(before[i].f, &(after[i].ux), &(after[i].uy), &(after[i].rho), i, params);

	int dir = 0;
	for (dir = 0; dir<9;dir += 1)
	{
		(after[i].f)[dir] = (before[i].f)[dir] + omega
			* (accel_gen(dir, after[i].ux, after[i].uy,
				after[i].ux * after[i].ux + after[i].uy
				* after[i].uy, after[i].rho, d2q9) - (before[i].f)[dir]);
	}
	return;
}

__device__
void doLeftWall(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params)
{
	(after[getIndex(x, y, params)].f)[dE] = d2q9[dE].wt  * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dNE] = d2q9[dNE].wt * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dSE] = d2q9[dSE].wt * (1 + 3 * v + 3 * v * v);
}

__device__
void doRightWall(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params)
{
	(after[getIndex(x, y, params)].f)[dW] = d2q9[dW].wt  * (1 - 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dNW] = d2q9[dNW].wt * (1 - 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dSW] = d2q9[dSW].wt * (1 - 3 * v + 3 * v * v);
}

//(top and bottom walls)
__device__
void doFlanks(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params)
{
	(after[getIndex(x, y, params)].f)[d0] = d2q9[d0].wt  * (1 - 1.5 * v * v);
	(after[getIndex(x, y, params)].f)[dE] = d2q9[dE].wt  * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dW] = d2q9[dW].wt  * (1 - 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dN] = d2q9[dN].wt  * (1 - 1.5 * v * v);
	(after[getIndex(x, y, params)].f)[dS] = d2q9[dS].wt  * (1 - 1.5 * v * v);
	(after[getIndex(x, y, params)].f)[dNE] = d2q9[dNE].wt * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dSE] = d2q9[dSE].wt * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dNW] = d2q9[dNW].wt * (1 - 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dSW] = d2q9[dSW].wt * (1 - 3 * v + 3 * v * v);
}

__device__
void streamEdgeCases(int x, int y, lbm_node* after, unsigned char* barrier,
	parameter_set* params, d2q9_node* d2q9)
{

	if (x == 0)
	{
		if (barrier[getIndex(x, y, params)] != 1)
		{
			doLeftWall(x, y, after, d2q9, params->v, params);
		}
	}
	else if (x == params->width - 1)
	{
		if (barrier[getIndex(x, y, params)] != 1)
		{
			doRightWall(x, y, after, d2q9, params->v, params);
		}
	}
	else if (y == 0 || y == params->width - 1)
	{
		if (barrier[getIndex(x, y, params)] != 1)
		{
			doFlanks(x, y, after, d2q9, params->v, params);
		}
	}
}

//stream: handle particle propagation, ignoring edge cases.
__global__
void stream(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
	unsigned char* barrier, parameter_set* params)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = getIndex(x, y, params);


	//toss out out of bounds and edge cases
	if (x < 0 || x >= params->width || y < 0 || y >= params->height)
		return;

	after[i].rho = before[i].rho;
	after[i].ux = before[i].ux;
	after[i].uy = before[i].uy;

	if (!(x > 0 && x < params->width - 1 && y > 0 && y < params->height - 1))
	{
		//return;
		streamEdgeCases(x, y, after, barrier, params, d2q9);
	}
	else
	{
		//propagate all f values around a bit
		int dir = 0;
		for (dir = 0;dir < 9;dir += 1)
		{
			(after[getIndex(d2q9[dir].ex + x, -d2q9[dir].ey + y, params)].f)[dir] =
				before[i].f[dir];
		}
	}
}

__global__
void bounceAndRender(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
	unsigned char* barrier, parameter_set* params, uchar4* image, int prex, int prey)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = getIndex(x, y, params);

	//toss out out of bounds and edge cases
	if (x < 0 || x >= params->width || y < 0 || y >= params->height)
		return;

	if (x > 0 && x < params->width - 1 && y>0 && y < params->height - 1)
	{
		if (barrier[i] == 1)
		{
			int dir;
			for (dir = 1; dir < 9; dir += 1)
			{
				if (d2q9[dir].op > 0 && (after[i].f)[dir]>0)
				{
					(after[getIndex(d2q9[dir].ex + x, -d2q9[dir].ey + y, params)].f)[dir]
						= (before[i].f)[d2q9[dir].op];
				}
			}
		}
	}

	computeColor(after, x, y, params, image, barrier, prex, prey);
}

//determine front and back lattice buffer orientation
//and launch all 3 LBM kernels
extern "C" void kernelLauncher(uchar4* image)
{
	if (needsUpdate)
	{
		cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char)*params.width * params.height, cudaMemcpyHostToDevice);
		cudaMemcpy(params_gpu, &params, sizeof(params), cudaMemcpyHostToDevice);
		needsUpdate = 0;
		cudaDeviceSynchronize(); // Wait for the GPU to finish
	}

	lbm_node* before = array1_gpu;
	lbm_node* after = array2_gpu;

	//determine number of threads and blocks required
	dim3 threads_per_block = dim3(32, 32, 1);
	dim3 number_of_blocks = dim3(params.width / 32 + 1, params.height / 32 + 1, 1);

	collide<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, before, after, params_gpu, barrier_gpu);

	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish

	before = array2_gpu;
	after = array1_gpu;

	stream<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, before, after, barrier_gpu, params_gpu);

	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish

	bounceAndRender<<<number_of_blocks, threads_per_block >>>(d2q9_gpu, before, after, barrier_gpu, params_gpu, image, prex, prey);

	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
}
