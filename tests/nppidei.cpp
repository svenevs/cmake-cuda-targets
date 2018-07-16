////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271062/#5271062
////////////////////////////////////////////////////////////////////////////////
#include <nppi_data_exchange_and_initialization.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * 8-bit image copy.
     */
    const int imgrows = 32;
    const int imgcols = 32;
    Npp8s *d_pSrc, *d_pDst;
    NppiSize oSizeROI;  oSizeROI.width = imgcols;  oSizeROI.height = imgrows;
    const int imgsize = imgrows*imgcols*sizeof(d_pSrc[0]);
    const int imgpix  = imgrows*imgcols;
    const int nSrcStep = imgcols*sizeof(d_pSrc[0]);
    const int nDstStep = imgcols*sizeof(d_pDst[0]);
    const int pixval = 1;
    cudaError_t err = cudaMalloc((void **)&d_pSrc, imgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pDst, imgsize);
    assert(err == cudaSuccess);
    // set image to pixval initially
    err = cudaMemset(d_pSrc, pixval, imgsize);
    assert(err == cudaSuccess);
    err = cudaMemset(d_pDst, 0, imgsize);
    assert(err == cudaSuccess);
    // copy src to dst
    NppStatus ret =  nppiCopy_8s_C1R(d_pSrc, nSrcStep, d_pDst, nDstStep, oSizeROI);
    assert(ret == NPP_NO_ERROR);
    Npp8s *h_imgres = new Npp8s[imgpix];
    err = cudaMemcpy(h_imgres, d_pDst, imgsize, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    // test that dst = pixval
    for (int i = 0; i < imgpix; i++) assert(h_imgres[i] == pixval);

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
