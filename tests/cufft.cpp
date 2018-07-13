//////////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, IT IS THE PROPERTY OF NVIDIA.  Comes from:        //
// https://docs.nvidia.com/cuda/cufft/index.html#oned-complex-to-complex-transforms //
//////////////////////////////////////////////////////////////////////////////////////

// i also dont know if this code does anything meaningful, but if it links i consider it a win ....

#include <cufft.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define NX 256
#define BATCH 1

int main(void) {
    cufftHandle plan;
    cufftComplex *data;
    cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
    if (cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return 1;
    }

    if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return 1;
    }

    // ...

    /* Note:
     *  Identical pointers to input and output arrays implies in-place transformation
     */

    if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
        return 1;
    }

    if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
        return 1;
    }

    /*
     *  Results may not be immediately available so block device until all
     *  tasks have completed
     */

    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
        return 1;
    }

    /*
     *  Divide by number of elements in data set to get back original data
     */

    // ...

    cufftDestroy(plan);
    cudaFree(data);

    printf("Everything seems to have executed correctly.\n");

    return 0;
}
