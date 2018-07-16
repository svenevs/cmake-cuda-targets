////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271053/#5271053
////////////////////////////////////////////////////////////////////////////////

#include <nppi_color_conversion.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.
     */

    const int imgrows = 32;
    const int imgcols = 32;
    Npp8u *d_pSrc, *d_pDst;
    const int pixval = 9;
    float R, G, B;
    R = B = G = pixval;
    const int nGray =  (int)(0.299F * R + 0.587F * G + 0.114F * B);
    NppiSize oSizeROI;  oSizeROI.width = imgcols;  oSizeROI.height = imgrows;
    const int srcimgsize = imgrows*imgcols*3*sizeof(Npp8u);
    const int dstimgsize = imgrows*imgcols*sizeof(Npp8u);
    const int imgpix  = imgrows*imgcols;
    const int nSrcStep = imgcols*3*sizeof(Npp8u);
    const int nDstStep = imgcols*sizeof(Npp8u);
    cudaError_t err = cudaMalloc((void **)&d_pSrc, srcimgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pDst, dstimgsize);
    assert(err == cudaSuccess);
    // set image (all components) to pixval initially
    err = cudaMemset(d_pSrc, pixval, srcimgsize);
    assert(err == cudaSuccess);
    // convert image to gray
    NppStatus ret = nppiRGBToGray_8u_C3C1R(d_pSrc, nSrcStep, d_pDst, nDstStep, oSizeROI);
    assert(ret == NPP_NO_ERROR);
    Npp8u *h_imgres = new Npp8u[imgpix];
    err = cudaMemcpy(h_imgres, d_pDst, dstimgsize, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    // test result
    for (int i = 0; i < imgpix; i++) assert(h_imgres[i] == nGray);

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
