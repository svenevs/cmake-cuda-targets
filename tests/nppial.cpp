////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271051/#5271051
////////////////////////////////////////////////////////////////////////////////
#include <nppi_arithmetic_and_logical_operations.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>

int main(){
    /**
     * One 8-bit unsigned char channel in place image add constant, scale, then clamp to saturated value.
     */
    const int imgrows = 32;
    const int imgcols = 32;
    const Npp8u nConstant = 4;
    Npp8u *d_pSrcDst;
    const int nScaleFactor = 2;
    const int nResult = nConstant >> nScaleFactor;
    NppiSize oSizeROI;  oSizeROI.width = imgcols;  oSizeROI.height = imgrows;
    const int imgsize = imgrows*imgcols*sizeof(d_pSrcDst[0]);
    const int imgpix  = imgrows*imgcols;
    const int nSrcDstStep = imgcols*sizeof(d_pSrcDst[0]);
    cudaError_t err = cudaMalloc((void **)&d_pSrcDst, imgsize);
    assert(err == cudaSuccess);
    // set image to 0 initially
    err = cudaMemset(d_pSrcDst, 0, imgsize);
    assert(err == cudaSuccess);
    // add nConstant to each pixel, then multiply each pixel by 2^-nScaleFactor
    NppStatus ret =  nppiAddC_8u_C1IRSfs(nConstant, d_pSrcDst, nSrcDstStep, oSizeROI, nScaleFactor);
    assert(ret == NPP_NO_ERROR);
    Npp8u *h_imgres = new Npp8u[imgpix];
    err = cudaMemcpy(h_imgres, d_pSrcDst, imgsize, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    // test that result = nConstant * 2^-nScaleFactor
    for (int i = 0; i < imgpix; i++) assert(h_imgres[i] == nResult);

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
