////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271073/#5271073
////////////////////////////////////////////////////////////////////////////////
#include <nppi_morphological_operations.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * Single-channel 8-bit unsigned integer 3x3 dilation.
     */
    const int simgrows = 32;
    const int simgcols = 32;
    const int maxval = 5;
    Npp8u *d_pSrc, *d_pDst;
    NppiSize oROI;  oROI.width = simgcols-2;  oROI.height = simgrows-2;
    const int simgsize = simgrows*simgcols*sizeof(d_pSrc[0]);
    const int dimgsize = oROI.width*oROI.height*sizeof(d_pSrc[0]);
    const int simgpix  = simgrows*simgcols;
    const int dimgpix  = oROI.width*oROI.height;
    const int nSrcStep = simgcols*sizeof(d_pSrc[0]);
    const int nDstStep = oROI.width*sizeof(d_pDst[0]);
    Npp8u *h_img = new Npp8u[simgpix];
    for (int i = 0; i < simgpix; i++) h_img[i] = (i%2)*maxval;
    cudaError_t err = cudaMalloc((void **)&d_pSrc, simgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pDst, dimgsize);
    assert(err == cudaSuccess);
    err = cudaMemcpy(d_pSrc, h_img, simgsize, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    err = cudaMemset(d_pDst, 0, dimgsize);
    assert(err == cudaSuccess);
    // do 3x3 max finding
    NppStatus ret =  nppiDilate3x3_8u_C1R(d_pSrc+simgrows+1, nSrcStep, d_pDst, nDstStep, oROI);
    assert(ret == NPP_NO_ERROR);
    err = cudaMemcpy(h_img, d_pDst, dimgsize, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    // test for alll pixels at maxval
    for (int i = 0; i < dimgpix; i++) assert(h_img[i] == maxval);

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
