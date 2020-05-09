#include <helper_gl.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "lbm_delegate.cuh"
#include "lbm_global.cuh"

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

void lbm_delegate::initPboResource(GLuint pbo)
{
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void lbm_delegate::initCUDA(d2q9_node* d2q9, lbm_node* array1, lbm_node* array2, unsigned char* barrier)
{
	cudaError_t ce = cudaMalloc(&d2q9_gpu, 9 * sizeof(d2q9_node));
	ce = cudaMalloc(&barrier_gpu, sizeof(unsigned char) * LATTICE_DIMENSION);
	ce = cudaMalloc(&array1_gpu, sizeof(lbm_node) * LATTICE_DIMENSION);
	ce = cudaMalloc(&array2_gpu, sizeof(lbm_node) * LATTICE_DIMENSION);

	ce = cudaMemcpy(d2q9_gpu, d2q9, sizeof(d2q9_node) * 9, cudaMemcpyHostToDevice);
	ce = cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
	ce = cudaMemcpy(array1_gpu, array1, sizeof(lbm_node) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);
	ce = cudaMemcpy(array2_gpu, array2, sizeof(lbm_node) * LATTICE_DIMENSION, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
}

void lbm_delegate::launchKernels(render_mode mode, bool barriersUpdated, unsigned char* barrier)
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

		color<<<number_of_blocks, threads_per_block>>>(mode, array1_gpu, d_out, barrier);
		cudaDeviceSynchronize();
	}

	//unmap the resources for next time
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void lbm_delegate::freeCUDA()
{
	cudaFree(d2q9_gpu);
	cudaFree(array1_gpu);
	cudaFree(array2_gpu);
	cudaFree(barrier_gpu);
	cudaGraphicsUnregisterResource(cuda_pbo_resource);
}
