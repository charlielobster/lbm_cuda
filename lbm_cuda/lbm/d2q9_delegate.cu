#include <helper_gl.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "d2q9_delegate.h"
#include "d2q9_global.cuh"

void d2q9_delegate::resetLattice(GLuint pbo)
{
	barrier = (unsigned char*)calloc(LATTICE_DIMENSION, sizeof(unsigned char));

	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

	array2 = (d2q9_lbm_node*)calloc(LATTICE_DIMENSION, sizeof(d2q9_lbm_node));

	d2q9_velocity_set* d2q9 = (d2q9_velocity_set*)calloc(9, sizeof(d2q9_velocity_set));

	d2q9[0].x_position = 0;		d2q9[0].y_position = 0;		d2q9[0].weight = 4.0 / 9.0;		d2q9[0].opposite = 0;
	d2q9[1].x_position = 1;		d2q9[1].y_position = 0;		d2q9[1].weight = 1.0 / 9.0;		d2q9[1].opposite = 3;
	d2q9[2].x_position = 0;		d2q9[2].y_position = 1;		d2q9[2].weight = 1.0 / 9.0;		d2q9[2].opposite = 4;
	d2q9[3].x_position = -1;	d2q9[3].y_position = 0;		d2q9[3].weight = 1.0 / 9.0;		d2q9[3].opposite = 1;
	d2q9[4].x_position = 0;		d2q9[4].y_position = -1;	d2q9[4].weight = 1.0 / 9.0;		d2q9[4].opposite = 2;
	d2q9[5].x_position = 1;		d2q9[5].y_position = 1;		d2q9[5].weight = 1.0 / 36.0;	d2q9[5].opposite = 7;
	d2q9[6].x_position = -1;	d2q9[6].y_position = 1;		d2q9[6].weight = 1.0 / 36.0;	d2q9[6].opposite = 8;
	d2q9[7].x_position = -1;	d2q9[7].y_position = -1;	d2q9[7].weight = 1.0 / 36.0;	d2q9[7].opposite = 5;
	d2q9[8].x_position = 1;		d2q9[8].y_position = -1;	d2q9[8].weight = 1.0 / 36.0;	d2q9[8].opposite = 6;

	//out = (unsigned char*)calloc(LATTICE_DIMENSION, sizeof(unsigned char));
	array1 = (d2q9_lbm_node*)calloc(LATTICE_DIMENSION, sizeof(d2q9_lbm_node));	
	int i;
	for (int x = 0; x < LATTICE_WIDTH; x++)
	{
		for (int y = 0; y < LATTICE_HEIGHT; y++)
		{
			i = D2Q9_INDEX(x, y);
			array1[i].vectors[ZERO] = d2q9[ZERO].weight * (1 - 1.5 * 1e-2);
			array1[i].vectors[EAST] = d2q9[EAST].weight * (1 + 3e-1+ 3e-2);
			array1[i].vectors[WEST] = d2q9[WEST].weight * (1 - 3e-1+ 3e-2);
			array1[i].vectors[NORTH] = d2q9[NORTH].weight * (1 - 1.5 * 1e-2);
			array1[i].vectors[SOUTH] = d2q9[SOUTH].weight * (1 - 1.5 * 1e-2);
			array1[i].vectors[NORTHEAST] = d2q9[NORTHEAST].weight * (1 + 3e-1+ 3e-2);
			array1[i].vectors[SOUTHEAST] = d2q9[SOUTHEAST].weight * (1 + 3e-1+ 3e-2);
			array1[i].vectors[NORTHWEST] = d2q9[NORTHWEST].weight * (1 - 3e-1+ 3e-2);
			array1[i].vectors[SOUTHWEST] = d2q9[SOUTHWEST].weight * (1 - 3e-1+ 3e-2);
			array1[i].rho = 1;
			array1[i].ux = 1e-1;
			array1[i].uy = 0;
		}
	}

	cudaError_t ce = cudaMalloc(&d2q9_gpu, 9 * sizeof(d2q9_lbm_node));
	ce = cudaMalloc(&barrier_gpu, sizeof(unsigned char) * LATTICE_DIMENSION);
	ce = cudaMalloc(&array1_gpu, sizeof(d2q9_lbm_node) * LATTICE_DIMENSION);
	ce = cudaMalloc(&array2_gpu, sizeof(d2q9_lbm_node) * LATTICE_DIMENSION);

	ce = cudaMemcpy(d2q9_gpu, d2q9, sizeof(d2q9_velocity_set) * 9, cudaMemcpyHostToDevice);
	ce = cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
	ce = cudaMemcpy(array1_gpu, array1, sizeof(d2q9_lbm_node) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
	ce = cudaMemcpy(array2_gpu, array2, sizeof(d2q9_lbm_node) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
}

void d2q9_delegate::launchKernels(lbm_render_mode mode)
{
	//reset image pointer
	uchar4* d_out = 0;

	//set d_out as a texture memory pointer
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_out, NULL, cuda_pbo_resource);

	//launch cuda kernels to calculate LBM step
	for (int i = 0; i < 10; i++)
	{
		if (barrierUpdated)
		{
			cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize(); // Wait for the GPU to finish
		}

		//determine number of threads and blocks required
		dim3 threads_per_block = dim3(32, 32, 1);
		dim3 number_of_blocks = dim3(LATTICE_WIDTH / 32 + 1, LATTICE_HEIGHT / 32 + 1, 1);

		d2q9_collide<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, array1_gpu, array2_gpu, barrier_gpu);
		cudaDeviceSynchronize();

		d2q9_stream<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, array2_gpu, array1_gpu, barrier_gpu);
		cudaDeviceSynchronize();

		d2q9_bounce<<<number_of_blocks, threads_per_block>>>(d2q9_gpu, array2_gpu, array1_gpu, barrier_gpu, d_out);
		cudaDeviceSynchronize();	

		color<<<number_of_blocks, threads_per_block>>>(mode, array1_gpu, d_out, barrier_gpu);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(out, d_out, sizeof(unsigned char) * LATTICE_DIMENSION, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	//unmap the resources for next time
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
	barrierUpdated = false;
}

void d2q9_delegate::clearBarrier()
{
	for (int i = 0; i < LATTICE_WIDTH; i++)
	{
		for (int j = 0; j < LATTICE_HEIGHT; j++)
		{
			barrier[D2Q9_INDEX(i, j)] = 0;
		}
	}
	barrierUpdated = true;
}

void d2q9_delegate::drawLineDiagonal()
{
	clearBarrier();
	for (int i = 0; i < LATTICE_HEIGHT / 4; i++)
	{

		barrier[D2Q9_INDEX((LATTICE_WIDTH / 3) + (i / 3), LATTICE_HEIGHT / 3 + i)] = 1;
	}
}

void d2q9_delegate::drawSquare()
{
	clearBarrier();
	for (int i = 0; i < LATTICE_HEIGHT / 4; i++)
	{
		for (int j = 0; j < LATTICE_HEIGHT / 4; j++)
		{
			barrier[D2Q9_INDEX(i + LATTICE_WIDTH / 3, j + LATTICE_HEIGHT * 3 / 8)] = 1;
		}
	}
}

void d2q9_delegate::freeCUDA()
{
	cudaFree(d2q9_gpu);
	cudaFree(array1_gpu);
	cudaFree(array2_gpu);
	cudaFree(barrier_gpu);
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
}