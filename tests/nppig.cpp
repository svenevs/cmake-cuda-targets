////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271071/#5271071
////////////////////////////////////////////////////////////////////////////////
#include <nppi_geometry_transforms.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * 1 channel 8-bit unsigned image mirror.
     */
    const int simgrows = 32;
    const int simgcols = 32;
    Npp8u *d_pSrc, *d_pDst;
    NppiSize oROI;  oROI.width = simgcols;  oROI.height = simgrows;
    const int simgsize = simgrows*simgcols*sizeof(d_pSrc[0]);
    const int dimgsize = oROI.width*oROI.height*sizeof(d_pSrc[0]);
    const int simgpix  = simgrows*simgcols;
    const int dimgpix  = oROI.width*oROI.height;
    const int nSrcStep = simgcols*sizeof(d_pSrc[0]);
    const int nDstStep = oROI.width*sizeof(d_pDst[0]);
    const NppiAxis flip = NPP_VERTICAL_AXIS;
    Npp8u *h_img = new Npp8u[simgpix];
    for (int i = 0; i < simgrows; i++)
        for (int j = 0; j < simgcols; j++)
            h_img[i*simgcols+j] = simgcols-j-1;
    cudaError_t err = cudaMalloc((void **)&d_pSrc, simgsize);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_pDst, dimgsize);
    assert(err == cudaSuccess);
    err = cudaMemcpy(d_pSrc, h_img, simgsize, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    // set image to pixval initially
    err = cudaMemset(d_pDst, 0, dimgsize);
    assert(err == cudaSuccess);
    // perform mirror op
    NppStatus ret =  nppiMirror_8u_C1R(d_pSrc, nSrcStep, d_pDst, nDstStep, oROI, flip);
    assert(ret == NPP_NO_ERROR);
    err = cudaMemcpy(h_img, d_pDst, dimgsize, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    // test for R to L flip
    for (int i = 0; i < oROI.height; i++)
        for (int j = 0; j < oROI.width; j++)
            assert(h_img[i*oROI.width+j] == j);

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
