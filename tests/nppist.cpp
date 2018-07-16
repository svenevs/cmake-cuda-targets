////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271074/#5271074
////////////////////////////////////////////////////////////////////////////////
#include <nppi_statistics_functions.h>
// note that functions from nppi_linear_transforms.h would be built/linked similarly
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * One-channel 8-bit unsigned image sum.
     */
    const int simgrows = 32;
    const int simgcols = 32;
    const int pixval = 1;
    Npp8u *d_pSrc, *d_pBuf;
    NppiSize oROI;  oROI.width = simgcols;  oROI.height = simgrows;
    const int simgsize = simgrows*simgcols*sizeof(d_pSrc[0]);
    const int simgpix  = simgrows*simgcols;
    const int nSrcStep = simgcols*sizeof(d_pSrc[0]);
    Npp64f *d_pSum, h_Sum;
    cudaError_t err = cudaMalloc((void **)&d_pSrc, simgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pBuf, 8*simgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pSum, sizeof(h_Sum));
    assert(err == cudaSuccess);
    err = cudaMemset(d_pSrc, pixval, simgsize);
    assert(err == cudaSuccess);
    // find sum of all pixels
    NppStatus ret =  nppiSum_8u_C1R(d_pSrc, nSrcStep, oROI, d_pBuf, d_pSum);
    assert(ret == NPP_NO_ERROR);
    err = cudaMemcpy(&h_Sum, d_pSum, sizeof(h_Sum), cudaMemcpyDeviceToHost);
    // test for proper sum
    assert(h_Sum == pixval*simgrows*simgcols);

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
