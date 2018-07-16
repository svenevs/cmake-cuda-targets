////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271066/#5271066
////////////////////////////////////////////////////////////////////////////////
#include <nppi_filtering_functions.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * 8-bit unsigned single-channel 1D row convolution.
     */
    const int simgrows = 32;
    const int simgcols = 32;
    Npp8u *d_pSrc, *d_pDst;
    const int nMaskSize = 3;
    NppiSize oROI;  oROI.width = simgcols - nMaskSize;  oROI.height = simgrows;
    const int simgsize = simgrows*simgcols*sizeof(d_pSrc[0]);
    const int dimgsize = oROI.width*oROI.height*sizeof(d_pSrc[0]);
    const int simgpix  = simgrows*simgcols;
    const int dimgpix  = oROI.width*oROI.height;
    const int nSrcStep = simgcols*sizeof(d_pSrc[0]);
    const int nDstStep = oROI.width*sizeof(d_pDst[0]);
    const int pixval = 1;
    const int nDivisor = 1;
    const Npp32s h_pKernel[nMaskSize] = {pixval, pixval, pixval};
    Npp32s *d_pKernel;
    const Npp32s nAnchor = 2;
    cudaError_t err = cudaMalloc((void **)&d_pSrc, simgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pDst, dimgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pKernel, nMaskSize*sizeof(d_pKernel[0]));
    assert(err == cudaSuccess);
    // set image to pixval initially
    err = cudaMemset(d_pSrc, pixval, simgsize);
    assert(err == cudaSuccess);
    err = cudaMemset(d_pDst, 0, dimgsize);
    assert(err == cudaSuccess);
    err = cudaMemcpy(d_pKernel, h_pKernel, nMaskSize*sizeof(d_pKernel[0]), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    // copy src to dst
    NppStatus ret =  nppiFilterRow_8u_C1R(d_pSrc, nSrcStep, d_pDst, nDstStep, oROI, d_pKernel, nMaskSize, nAnchor, nDivisor);
    assert(ret == NPP_NO_ERROR);
    Npp8u *h_imgres = new Npp8u[dimgpix];
    err = cudaMemcpy(h_imgres, d_pDst, dimgsize, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    // test for filtering
    for (int i = 0; i < dimgpix; i++) assert(h_imgres[i] == (pixval*pixval*nMaskSize));

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
