#include <helper_gl.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "lbm_delegate.h"
#include "d2q9_global.cuh"

void lbm_delegate::printDeviceInfo()
{
	int nDevices = 0;
	cudaError_t ce = cudaGetDeviceCount(&nDevices);
	cudaDeviceProp prop;
	for (int i = 0; i < nDevices; ++i)
	{
		ce = cudaGetDeviceProperties(&prop, i);
		printf("Device number: %d\n", i);
		printf("Device name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
		printf("Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
		printf("Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);
		if (ce != cudaSuccess) { printf("error: %s\n", cudaGetErrorString(ce)); }
	}
}

void lbm_delegate::launchKernels(render_mode mode, bool barriersUpdated, unsigned char* barrier, unsigned char* out)
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

void lbm_delegate::resetLattice(GLuint pbo, unsigned char* barrier)
{
	initPboResource(pbo);
	a2 = (d2q9_lbm_node*)calloc(LATTICE_DIMENSION, sizeof(d2q9_lbm_node));
	d2q9_position_weight* d2q9 = (d2q9_position_weight*)calloc(9, sizeof(d2q9_position_weight));
	initD2q9(d2q9);
	initA1(d2q9);	
	initCUDA(d2q9, a1, a2, barrier);	
}

void lbm_delegate::freeCUDA()
{
	cudaFree(d2q9_gpu);
	cudaFree(array1_gpu);
	cudaFree(array2_gpu);
	cudaFree(barrier_gpu);
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
}

void lbm_delegate::initPboResource(GLuint pbo)
{
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

//provide LBM constants for d2q9 style nodes
//assumes positive is up and right, whereas our program assumes positive down and right.
void lbm_delegate::initD2q9(d2q9_position_weight* d2q9)
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

void lbm_delegate::initA1(d2q9_position_weight* d2q9)
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

void lbm_delegate::initCUDA(d2q9_position_weight* d2q9, d2q9_lbm_node* array1, d2q9_lbm_node* array2, unsigned char* barrier)
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

