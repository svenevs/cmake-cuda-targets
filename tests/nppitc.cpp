////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271081/#5271081
////////////////////////////////////////////////////////////////////////////////
#include <nppi_threshold_and_compare_operations.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * 1 channel 8-bit unsigned char image compare.
     * Compare pSrc1's pixels with corresponding pixels in pSrc2.
     */
    const int simgrows = 32;
    const int simgcols = 32;
    const int pixval = 1;
    Npp8u *d_pSrc1, *d_pSrc2, *d_pDst;
    NppiSize oROI;  oROI.width = simgcols;  oROI.height = simgrows;
    const int simgsize = simgrows*simgcols*sizeof(d_pSrc1[0]);
    const int simgpix  = simgrows*simgcols;
    const int nSrcStep = simgcols*sizeof(d_pSrc1[0]);
    cudaError_t err = cudaMalloc((void **)&d_pSrc1, simgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pSrc2, simgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pDst, simgsize);
    assert(err == cudaSuccess);
    err = cudaMemset(d_pSrc1, 0, simgsize);
    assert(err == cudaSuccess);
    err = cudaMemset(d_pSrc2, pixval, simgsize);
    assert(err == cudaSuccess);
    err = cudaMemset(d_pDst, 0, simgsize);
    assert(err == cudaSuccess);
    NppCmpOp eCompOp = NPP_CMP_LESS;
    // compare images
    NppStatus ret =  nppiCompare_8u_C1R(d_pSrc1, nSrcStep, d_pSrc2, nSrcStep, d_pDst,  nSrcStep, oROI, eCompOp);
    assert(ret == NPP_NO_ERROR);
    Npp8u *h_img = new Npp8u[simgpix];
    err = cudaMemcpy(h_img, d_pDst, simgsize, cudaMemcpyDeviceToHost);
    // test for proper compare
    for (int i = 0; i < simgpix; i++) assert(h_img[i]);

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
