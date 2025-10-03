#include <helper_gl.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "d2q9_delegate.h"

void d2q9_delegate::launchKernels(render_mode mode, bool barriersUpdated, unsigned char* barrier, unsigned char* out)
{
	//reset image pointer
	uchar4* d_out = 0;

	//set d_out as a texture memory pointer
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_out, NULL, cuda_pbo_resource);

	//launch cuda kernels to calculate LBM step
	for (int i = 0; i < STEPS_PER_RENDER; i++)
	{
		if (barriersUpdated)
		{
			cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize(); // Wait for the GPU to finish
		}

		//determine number of threads and blocks required
		dim3 threads_per_block = dim3(32, 32, 1);
		dim3 number_of_blocks = dim3(LATTICE_WIDTH / 32 + 1, LATTICE_HEIGHT / 32 + 1, 1);

		collide<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, array1_gpu, array2_gpu, barrier_gpu);
		cudaDeviceSynchronize();

		stream<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, array2_gpu, array1_gpu, barrier_gpu);
		cudaDeviceSynchronize();

		bounce<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, array2_gpu, array1_gpu, barrier_gpu, d_out);
		cudaDeviceSynchronize();	

		color<<<number_of_blocks, threads_per_block>>>(mode, array1_gpu, d_out, barrier_gpu);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(out, d_out, sizeof(unsigned char) * LATTICE_DIMENSION, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	//unmap the resources for next time
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void d2q9_delegate::resetLattice(GLuint pbo, unsigned char* barrier)
{
	initPboResource(pbo);
	a2 = (d2q9_lbm_node*)calloc(LATTICE_DIMENSION, sizeof(d2q9_lbm_node));
	d2q9_position_weight* d2q9 = (d2q9_position_weight*)calloc(9, sizeof(d2q9_position_weight));
	initD2q9(d2q9);
	initA1(d2q9);	
	initCUDA(d2q9, a1, a2, barrier);	
}

void d2q9_delegate::freeCUDA()
{
	cudaFree(d2q9_gpu);
	cudaFree(array1_gpu);
	cudaFree(array2_gpu);
	cudaFree(barrier_gpu);
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
}

void d2q9_delegate::initPboResource(GLuint pbo)
{
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

//provide LBM constants for d2q9 style nodes
//assumes positive is up and right, whereas our program assumes positive down and right.
void d2q9_delegate::initD2q9(d2q9_position_weight* d2q9)
{
	d2q9[0].x_position = 0;		d2q9[0].y_position = 0;		d2q9[0].weight = 4.0 / 9.0;		d2q9[0].opposite = 0;
	d2q9[1].x_position = 1;		d2q9[1].y_position = 0;		d2q9[1].weight = 1.0 / 9.0;		d2q9[1].opposite = 3;
	d2q9[2].x_position = 0;		d2q9[2].y_position = 1;		d2q9[2].weight = 1.0 / 9.0;		d2q9[2].opposite = 4;
	d2q9[3].x_position = -1;	d2q9[3].y_position = 0;		d2q9[3].weight = 1.0 / 9.0;		d2q9[3].opposite = 1;
	d2q9[4].x_position = 0;		d2q9[4].y_position = -1;	d2q9[4].weight = 1.0 / 9.0;		d2q9[4].opposite = 2;
	d2q9[5].x_position = 1;		d2q9[5].y_position = 1;		d2q9[5].weight = 1.0 / 36.0;	d2q9[5].opposite = 7;
	d2q9[6].x_position = -1;	d2q9[6].y_position = 1;		d2q9[6].weight = 1.0 / 36.0;	d2q9[6].opposite = 8;
	d2q9[7].x_position = -1;	d2q9[7].y_position = -1;	d2q9[7].weight = 1.0 / 36.0;	d2q9[7].opposite = 5;
	d2q9[8].x_position = 1;		d2q9[8].y_position = -1;	d2q9[8].weight = 1.0 / 36.0;	d2q9[8].opposite = 6;
}

void d2q9_delegate::initA1(d2q9_position_weight* d2q9)
{
	//out = (unsigned char*)calloc(LATTICE_DIMENSION, sizeof(unsigned char));
	a1 = (d2q9_lbm_node*)calloc(LATTICE_DIMENSION, sizeof(d2q9_lbm_node));	
	int i;
	for (int x = 0; x < LATTICE_WIDTH; x++)
	{
		for (int y = 0; y < LATTICE_HEIGHT; y++)
		{
			i = INDEX(x, y);
			a1[i].vectors[ZERO] = d2q9[ZERO].weight * (1 - 1.5 * VELOCITY_SQUARED);
			a1[i].vectors[EAST] = d2q9[EAST].weight * (1 + _3V+ _3V2);
			a1[i].vectors[WEST] = d2q9[WEST].weight * (1 - _3V+ _3V2);
			a1[i].vectors[NORTH] = d2q9[NORTH].weight * (1 - 1.5 * VELOCITY_SQUARED);
			a1[i].vectors[SOUTH] = d2q9[SOUTH].weight * (1 - 1.5 * VELOCITY_SQUARED);
			a1[i].vectors[NORTHEAST] = d2q9[NORTHEAST].weight * (1 + _3V+ _3V2);
			a1[i].vectors[SOUTHEAST] = d2q9[SOUTHEAST].weight * (1 + _3V+ _3V2);
			a1[i].vectors[NORTHWEST] = d2q9[NORTHWEST].weight * (1 - _3V+ _3V2);
			a1[i].vectors[SOUTHWEST] = d2q9[SOUTHWEST].weight * (1 - _3V+ _3V2);
			a1[i].rho = 1;
			a1[i].ux = VELOCITY;
			a1[i].uy = 0;
		}
	}
}

void d2q9_delegate::initCUDA(d2q9_position_weight* d2q9, d2q9_lbm_node* array1, d2q9_lbm_node* array2, unsigned char* barrier)
{
	cudaError_t ce = cudaMalloc(&d2q9_gpu, 9 * sizeof(d2q9_lbm_node));
	ce = cudaMalloc(&barrier_gpu, sizeof(unsigned char) * LATTICE_DIMENSION);
	ce = cudaMalloc(&array1_gpu, sizeof(d2q9_lbm_node) * LATTICE_DIMENSION);
	ce = cudaMalloc(&array2_gpu, sizeof(d2q9_lbm_node) * LATTICE_DIMENSION);

	ce = cudaMemcpy(d2q9_gpu, d2q9, sizeof(d2q9_position_weight) * 9, cudaMemcpyHostToDevice);
	ce = cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
	ce = cudaMemcpy(array1_gpu, array1, sizeof(d2q9_lbm_node) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
	ce = cudaMemcpy(array2_gpu, array2, sizeof(d2q9_lbm_node) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
}


__global__
static void collide(d2q9_position_weight* d2q9, d2q9_lbm_node* before, d2q9_lbm_node* after, unsigned char* barrier)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = INDEX(x, y);

	float omega = 1 / (3 * VISCOSITY + 0.5);

	//toss out out of bounds
	if (x < 0 || x >= LATTICE_WIDTH || y < 0 || y >= LATTICE_HEIGHT)
		return;

	macroGen(before[i].vectors, &(after[i].ux), &(after[i].uy), &(after[i].rho), i);

	for (int v = 0; v < 9; v += 1)
	{
		after[i].vectors[v] = before[i].vectors[v] + omega
			* (accelGen(v, after[i].ux, after[i].uy,
				after[i].ux * after[i].ux + after[i].uy
				* after[i].uy, after[i].rho, d2q9) - before[i].vectors[v]);
	}
	return;
}

//stream: handle particle propagation, ignoring edge cases.
__global__
static void stream(d2q9_position_weight* d2q9, d2q9_lbm_node* before, d2q9_lbm_node* after, unsigned char* barrier)
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
		streamEdgeCases(x, y, after, barrier, d2q9);
	}
	else
	{
		//propagate all f values around a bit
		for (int v = 0; v < 9; v += 1)
		{
			after[INDEX(d2q9[v].x_position + x, -d2q9[v].y_position + y)].vectors[v] =
				before[i].vectors[v];
		}
	}
}

__global__
static void bounce(d2q9_position_weight* d2q9, d2q9_lbm_node* before, d2q9_lbm_node* after,
		unsigned char* barrier, uchar4* image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = INDEX(x, y);

	if (x > 0 && x < LATTICE_WIDTH - 1 && y > 0 && y < LATTICE_HEIGHT - 1)
	{
		if (barrier[i] == 1)
		{
			for (int v = 1; v < 9; v += 1)
			{
				if (d2q9[v].opposite > 0 && after[i].vectors[v] > 0)
				{
					after[INDEX(d2q9[v].x_position + x, -d2q9[v].y_position + y)].vectors[v]
						= (before[i].vectors)[d2q9[v].opposite];
				}
			}
		}
	}
}

__global__
static void color(render_mode mode, d2q9_lbm_node* array, uchar4* image, unsigned char* barrier)
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
			image[i] = getRgbCurl(x, y, array);
			break;
		case SPEED:
			image[i] = getRgbU(sqrt(array[i].ux * array[i].ux + array[i].uy * array[i].uy));
			break;
		case UX:
			image[i] = getRgbU(array[i].ux);
			break;
		case UY:
			image[i] = getRgbU(array[i].uy);
			break;
		default:
			break;
		}
	}
}

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