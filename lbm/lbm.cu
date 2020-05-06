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
parameter_set* params_gpu;
struct cudaGraphicsResource* cuda_pbo_resource;

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
uchar4 getRgbRho(float i, parameter_set* params)
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
uchar4 getRgbU(float i)
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
uchar4 getRgbCurl(int x, int y, lbm_node* array, parameter_set* params)
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
void computeColor(lbm_node* array, int x, int y, parameter_set* params, uchar4* image, unsigned char* barrier)
{
	int i = getIndex(x, y, params);
	int prei = getIndex(params->prex, params->prey, params);

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
			image[i] = getRgbRho(array[i].rho, params);
			break;
		case(mCurl):
			image[i] = getRgbCurl(x, y, array, params);
			break;
		case(mSpeed):
			image[i] = getRgbU(sqrt(array[i].ux * array[i].ux + array[i].uy * array[i].uy));
			break;
		case(mUx):
			image[i] = getRgbU(array[i].ux);
			break;
		case(mUy):
			image[i] = getRgbU(array[i].uy);
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
void macroGen(float* f, float* ux, float* uy, float* rho, int i, parameter_set* params)
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

__global__
void collide(d2q9_node* d2q9, lbm_node* before, lbm_node* after, parameter_set* params, unsigned char* barrier)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = getIndex(x, y, params);

	float omega = 1 / (3 * params->viscosity + 0.5);

	//toss out out of bounds
	if (x < 0 || x >= params->width || y < 0 || y >= params->height)
		return;

	macroGen(before[i].f, &(after[i].ux), &(after[i].uy), &(after[i].rho), i, params);

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
	unsigned char* barrier, parameter_set* params, uchar4* image)
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

	computeColor(after, x, y, params, image, barrier);
}

//display stats of all detected cuda capable devices, and return the number
extern "C"
int deviceQuery()
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
		printf("  Device name: %s\n", prop.name);
		printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("  Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
		printf("  Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
		printf("  Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);
		if (ierr != cudaSuccess) { printf("error: %s\n", cudaGetErrorString(ierr)); }
	}

	return nDevices;
}

extern "C"
void initPboResource(GLuint pbo)
{
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

extern "C" 
void initCUDA(d2q9_node * d2q9, parameter_set *params, int W, int H, 
	lbm_node* array1, lbm_node* array2, unsigned char* barrier)
{
	ierrSync = cudaMalloc(&d2q9_gpu, 9 * sizeof(d2q9_node));
	ierrSync = cudaMalloc(&params_gpu, sizeof(parameter_set));
	ierrSync = cudaMalloc(&barrier_gpu, sizeof(unsigned char) * W * H);
	ierrSync = cudaMalloc(&array1_gpu, sizeof(lbm_node) * W * H);
	ierrSync = cudaMalloc(&array2_gpu, sizeof(lbm_node) * W * H);

	ierrSync = cudaMemcpy(d2q9_gpu, d2q9, sizeof(d2q9_node) * 9, cudaMemcpyHostToDevice);
	ierrSync = cudaMemcpy(params_gpu, params, sizeof(*params), cudaMemcpyHostToDevice);
	ierrSync = cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char) * W * H, cudaMemcpyHostToDevice);
	ierrSync = cudaMemcpy(array1_gpu, array1, sizeof(lbm_node) * W * H, cudaMemcpyHostToDevice);
	ierrSync = cudaMemcpy(array2_gpu, array2, sizeof(lbm_node) * W * H, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
}

extern "C" 
void freeCUDA()
{
	cudaFree(d2q9_gpu);
	cudaFree(params_gpu);
	cudaFree(array1_gpu);
	cudaFree(array2_gpu);
	cudaFree(barrier_gpu);
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
}

//render the image (but do not display it yet)
extern "C"
void render(int delta_t, parameter_set* params, unsigned char* barrier)
{
	//reset image pointer
	uchar4* d_out = 0;

	//set d_out as a texture memory pointer
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_out, NULL, cuda_pbo_resource);

	//launch cuda kernels to calculate LBM step
	for (int i = 0; i < params->stepsPerRender; i++)
	{
		if (params->needsUpdate)
		{
			cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char) * params->width * params->height, cudaMemcpyHostToDevice);
			cudaMemcpy(params_gpu, params, sizeof(*params), cudaMemcpyHostToDevice);
			params->needsUpdate = 0;
			cudaDeviceSynchronize(); // Wait for the GPU to finish
		}

		lbm_node* before = array1_gpu;
		lbm_node* after = array2_gpu;

		//determine number of threads and blocks required
		dim3 threads_per_block = dim3(32, 32, 1);
		dim3 number_of_blocks = dim3(params->width / 32 + 1, params->height / 32 + 1, 1);

		collide<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, before, after, params_gpu, barrier_gpu);

		ierrSync = cudaGetLastError();
		ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish

		before = array2_gpu;
		after = array1_gpu;

		stream<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, before, after, barrier_gpu, params_gpu);

		ierrSync = cudaGetLastError();
		ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish

		bounceAndRender<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, before, after, barrier_gpu, params_gpu, d_out);

		ierrSync = cudaGetLastError();
		ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish	
	}

	//unmap the resources for next time
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}