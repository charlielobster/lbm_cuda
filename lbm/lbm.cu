#include <helper_gl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "lbm.cuh"

cudaError_t ierrAsync;
cudaError_t ierrSync;

lbm_node* array1_gpu;
lbm_node* array2_gpu;
unsigned char* barrier_gpu;
d2q9_node* d2q9_gpu;
struct cudaGraphicsResource* cuda_pbo_resource;

__device__
uchar4 getRgbU(float i)
{

	uchar4 val;
	if (i == i)
	{
		val.w = 255;
		val.x = 0;
		val.y = CLIP(i*255.0 / 1.0);
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
float computeCurlMiddleCase(int x, int y, lbm_node * array1) 
{
	return (array1[INDEX(x, y + 1)].ux - array1[INDEX(x, y - 1)].ux)
		- (array1[INDEX(x + 1, y)].uy - array1[INDEX(x - 1, y)].uy);
}

__device__
uchar4 getRgbCurl(int x, int y, lbm_node* array)
{
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

	if (array[INDEX(x, y)].rho != array[INDEX(x, y)].rho)
	{
		val.x = 255;
		val.y = 0;
		val.z = 0;
		val.w = 255;
	}
	return val;
}

__device__
void computeColor(renderMode mode, lbm_node* array, int x, int y, 
	uchar4* image, unsigned char* barrier)
{
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
		case(CURL):
			image[i] = getRgbCurl(x, y, array);
			break;
		case(SPEED):
			image[i] = getRgbU(sqrt(array[i].ux * array[i].ux + array[i].uy * array[i].uy));
			break;
		case(UX):
			image[i] = getRgbU(array[i].ux);
			break;
		case(UY):
			image[i] = getRgbU(array[i].uy);
			break;
		}
	}	
}

__device__
void macroGen(float* f, float* ux, float* uy, float* rho, int i)
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
float accelGen(int node_num, float ux, float uy, float u2, float rho, d2q9_node* d2q9)
{
	float u_direct = ux * d2q9[node_num].ex + uy * (-d2q9[node_num].ey);
	float unweighted = 1 + 3 * u_direct + 4.5*u_direct*u_direct - 1.5*u2;

	return rho * d2q9[node_num].wt * unweighted;
}

__device__
void doLeftWall(int x, int y, lbm_node* after, d2q9_node* d2q9)
{
	(after[INDEX(x, y)].f)[EAST] = d2q9[EAST].wt  * (1 + 3 * VELOCITY + 3 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[NORTHEAST] = d2q9[NORTHEAST].wt * (1 + 3 * VELOCITY + 3 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[SOUTHEAST] = d2q9[SOUTHEAST].wt * (1 + 3 * VELOCITY + 3 * VELOCITY_SQUARED);
}

__device__
void doRightWall(int x, int y, lbm_node* after, d2q9_node* d2q9)
{
	(after[INDEX(x, y)].f)[WEST] = d2q9[WEST].wt  * (1 - 3 * VELOCITY + 3 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[NORTHWEST] = d2q9[NORTHWEST].wt * (1 - 3 * VELOCITY + 3 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[SOUTHWEST] = d2q9[SOUTHWEST].wt * (1 - 3 * VELOCITY + 3 * VELOCITY_SQUARED);
}

//(top and bottom walls)
__device__
void doFlanks(int x, int y, lbm_node* after, d2q9_node* d2q9)
{
	(after[INDEX(x, y)].f)[NONE] = d2q9[NONE].wt  * (1 - 1.5 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[EAST] = d2q9[EAST].wt  * (1 + 3 * VELOCITY + 3 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[WEST] = d2q9[WEST].wt  * (1 - 3 * VELOCITY + 3 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[NORTH] = d2q9[NORTH].wt  * (1 - 1.5 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[SOUTH] = d2q9[SOUTH].wt  * (1 - 1.5 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[NORTHEAST] = d2q9[NORTHEAST].wt * (1 + 3 * VELOCITY + 3 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[SOUTHEAST] = d2q9[SOUTHEAST].wt * (1 + 3 * VELOCITY + 3 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[NORTHWEST] = d2q9[NORTHWEST].wt * (1 - 3 * VELOCITY + 3 * VELOCITY_SQUARED);
	(after[INDEX(x, y)].f)[SOUTHWEST] = d2q9[SOUTHWEST].wt * (1 - 3 * VELOCITY + 3 * VELOCITY_SQUARED);
}

__device__
void streamEdgeCases(int x, int y, lbm_node* after, unsigned char* barrier, d2q9_node* d2q9)
{

	if (x == 0)
	{
		if (barrier[INDEX(x, y)] != 1)
		{
			doLeftWall(x, y, after, d2q9);
		}
	}
	else if (x == LATTICE_WIDTH - 1)
	{
		if (barrier[INDEX(x, y)] != 1)
		{
			doRightWall(x, y, after, d2q9);
		}
	}
	else if (y == 0 || y == LATTICE_WIDTH - 1)
	{
		if (barrier[INDEX(x, y)] != 1)
		{
			doFlanks(x, y, after, d2q9);
		}
	}
}

__global__
void collide(d2q9_node* d2q9, lbm_node* before, lbm_node* after, unsigned char* barrier)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = INDEX(x, y);

	float omega = 1 / (3 * VISCOSITY + 0.5);

	//toss out out of bounds
	if (x < 0 || x >= LATTICE_WIDTH || y < 0 || y >= LATTICE_HEIGHT)
		return;

	macroGen(before[i].f, &(after[i].ux), &(after[i].uy), &(after[i].rho), i);

	int dir = 0;
	for (dir = 0; dir < 9; dir += 1)
	{
		(after[i].f)[dir] = (before[i].f)[dir] + omega
			* (accelGen(dir, after[i].ux, after[i].uy,
				after[i].ux * after[i].ux + after[i].uy
				* after[i].uy, after[i].rho, d2q9) - (before[i].f)[dir]);
	}
	return;
}

//stream: handle particle propagation, ignoring edge cases.
__global__
void stream(d2q9_node* d2q9, lbm_node* before, lbm_node* after, unsigned char* barrier)
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
		int dir = 0;
		for (dir = 0;dir < 9;dir += 1)
		{
			(after[INDEX(d2q9[dir].ex + x, -d2q9[dir].ey + y)].f)[dir] =
				before[i].f[dir];
		}
	}
}

__global__
void bounceAndColor(renderMode mode, d2q9_node* d2q9, 
	lbm_node* before, lbm_node* after,
	unsigned char* barrier, uchar4* image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = INDEX(x, y);

	//toss out out of bounds and edge cases
	if (x < 0 || x >= LATTICE_WIDTH || y < 0 || y >= LATTICE_HEIGHT)
		return;

	if (x > 0 && x < LATTICE_WIDTH - 1 && y > 0 && y < LATTICE_HEIGHT - 1)
	{
		if (barrier[i] == 1)
		{
			int dir;
			for (dir = 1; dir < 9; dir += 1)
			{
				if (d2q9[dir].op > 0 && (after[i].f)[dir] > 0)
				{
					(after[INDEX(d2q9[dir].ex + x, -d2q9[dir].ey + y)].f)[dir]
						= (before[i].f)[d2q9[dir].op];
				}
			}
		}
	}

	computeColor(mode, after, x, y, image, barrier);
}

//display stats of all detected cuda capable devices, and return the number
extern "C"
void printDeviceInfo()
{
	cudaDeviceProp prop;
	int nDevices = 1;
	cudaError_t ierr;

	ierr = cudaGetDeviceCount(&nDevices);

	int i = 0;
	for (i = 0; i < nDevices; ++i)
	{
		ierr = cudaGetDeviceProperties(&prop, i);
		printf("Device number: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
		printf("Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
		printf("Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);
		if (ierr != cudaSuccess) { printf("error: %s\n", cudaGetErrorString(ierr)); }
	}
}

extern "C"
void initPboResource(GLuint pbo)
{
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

extern "C" 
void initCUDA(d2q9_node * d2q9, lbm_node* array1, lbm_node* array2, unsigned char* barrier)
{
	ierrSync = cudaMalloc(&d2q9_gpu, 9 * sizeof(d2q9_node));
	ierrSync = cudaMalloc(&barrier_gpu, sizeof(unsigned char) * LATTICE_DIMENSION);
	ierrSync = cudaMalloc(&array1_gpu, sizeof(lbm_node) * LATTICE_DIMENSION);
	ierrSync = cudaMalloc(&array2_gpu, sizeof(lbm_node) * LATTICE_DIMENSION);

	ierrSync = cudaMemcpy(d2q9_gpu, d2q9, sizeof(d2q9_node) * 9, cudaMemcpyHostToDevice);
	ierrSync = cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
	ierrSync = cudaMemcpy(array1_gpu, array1, sizeof(lbm_node) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
	ierrSync = cudaMemcpy(array2_gpu, array2, sizeof(lbm_node) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
}

extern "C" 
void freeCUDA()
{
	cudaFree(d2q9_gpu);
	cudaFree(array1_gpu);
	cudaFree(array2_gpu);
	cudaFree(barrier_gpu);
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
}

//render the image (but do not display it yet)
extern "C"
void launchKernels(bool barriersUpdated, renderMode mode, unsigned char* barrier)
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

		lbm_node* before = array1_gpu;
		lbm_node* after = array2_gpu;

		//determine number of threads and blocks required
		dim3 threads_per_block = dim3(32, 32, 1);
		dim3 number_of_blocks = dim3(LATTICE_WIDTH / 32 + 1, LATTICE_HEIGHT / 32 + 1, 1);

		collide<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, before, after, barrier_gpu);

		ierrSync = cudaGetLastError();
		ierrAsync = cudaDeviceSynchronize();

		before = array2_gpu;
		after = array1_gpu;

		stream<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, before, after, barrier_gpu);

		ierrSync = cudaGetLastError();
		ierrAsync = cudaDeviceSynchronize();

		bounceAndColor<<<number_of_blocks, threads_per_block>>>(mode, d2q9_gpu, before, after, barrier_gpu, d_out);

		ierrSync = cudaGetLastError();
		ierrAsync = cudaDeviceSynchronize();	
	}

	//unmap the resources for next time
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}