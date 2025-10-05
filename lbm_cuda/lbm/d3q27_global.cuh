#ifndef _D3Q27_GLOBAL_H_
#define _D3Q27_GLOBAL_H_

#include "d3q27_device.cuh"

__global__
static void d3q27_collide(d3q27_velocity_set* d3q27, d3q27_lbm_node* before, d3q27_lbm_node* after, unsigned char* barrier)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int i = D3Q27_INDEX(x, y, z);

    float omega = 1 / (3 * 5e-3 + 0.5);

    //toss out out of bounds
    if (x < 0 || x >= LATTICE_WIDTH || y < 0 || y >= LATTICE_HEIGHT)
        return;

    d3q27_device::macroGen(before[i].vectors, &(after[i].ux), &(after[i].uy), &(after[i].uz), &(after[i].rho), i);

    for (int v = 0; v < 27; v += 1)
    {
        after[i].vectors[v] = before[i].vectors[v] + omega
            * (d3q27_device::accelGen(v, after[i].ux, after[i].uy, after[i].uz,
                after[i].ux * after[i].ux + after[i].uy
                * after[i].uy, after[i].rho, d3q27) - before[i].vectors[v]);
    }
    return;
}

//stream: handle particle propagation, ignoring edge cases.
__global__
static void d3q27_stream(d3q27_velocity_set* d3q27, d3q27_lbm_node* before, d3q27_lbm_node* after, unsigned char* barrier)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int i = D3Q27_INDEX(x, y, z);

    //toss out out of bounds and edge cases
    if (x < 0 || x >= LATTICE_WIDTH || y < 0 || y >= LATTICE_HEIGHT)
        return;

    after[i].rho = before[i].rho;
    after[i].ux = before[i].ux;
    after[i].uy = before[i].uy;

    if (!(x > 0 && x < LATTICE_WIDTH - 1 && y > 0 && y < LATTICE_HEIGHT - 1))
    {
        //return;
        d3q27_device::streamEdgeCases(x, y, z, after, barrier, d3q27);
    }
    else
    {
        //propagate all f values around a bit
        for (int v = 0; v < 27; v ++)
        {
        //    after[D3Q27_INDEX(d2q9[v].x_position + x, -d2q9[v].y_position + y)].vectors[v] =
           //     before[i].vectors[v];
        }
    }
}

__global__
static void d2q9_bounce(d3q27_velocity_set* d3q27, d3q27_lbm_node* before, d3q27_lbm_node* after,
        unsigned char* barrier, uchar4* image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int i = D3Q27_INDEX(x, y, z);

    if (x > 0 && x < LATTICE_WIDTH - 1 && y > 0 && y < LATTICE_HEIGHT - 1)
    {
        if (barrier[i] == 1)
        {
            for (int v = 1; v < 27; v ++)
            {
                if (d3q27[v].opposite > 0 && after[i].vectors[v] > 0)
                {
                 //   after[D3Q27_INDEX(d3q27[v].x_position + x, -d3q27[v].y_position + y)].vectors[v]
                 //       = (before[i].vectors)[d3q27[v].opposite];
                }
            }
        }
    }
}

__global__
static void d3q27_color(lbm_render_mode mode, d3q27_lbm_node* array, uchar4* image, unsigned char* barrier)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // toss out out of bounds and edge cases
    if (x < 0 || x >= LATTICE_WIDTH || y < 0 || y >= LATTICE_HEIGHT)
        return;

    int i = D3Q27_INDEX(x, y, z);

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
            image[i] = d3q27_device::getRgbCurl(x, y, z, array);
            break;
        case SPEED:
            image[i] = d3q27_device::getRgbU(sqrt(array[i].ux * array[i].ux + array[i].uy * array[i].uy + array[i].uz * array[i].uz));
            break;
        case UX:
            image[i] = d3q27_device::getRgbU(array[i].ux);
            break;
        case UY:
            image[i] = d3q27_device::getRgbU(array[i].uy);
            break;
        default:
            break;
        }
    }
}

#endif
