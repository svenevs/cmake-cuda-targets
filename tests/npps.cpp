////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271082/#5271082
////////////////////////////////////////////////////////////////////////////////
#include <npps.h>
// currently the above header file and this method also include use of:
//
// npps_arithmetic_and_logical_operations.h  npps_support_functions.h
// npps_conversion_functions.h               npps_initialization.h
// npps_filtering_functions.h                npps_statistics_functions.h
//
#include <cuda_runtime_api.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * 1 channel 8-bit unsigned char zero vector
     */
    const int len = 32;
    Npp8u *d_pDst;
    const int vsize = len*sizeof(d_pDst[0]);
    cudaError_t err = cudaMalloc((void **)&d_pDst, vsize);
    assert(err == cudaSuccess);
    err = cudaMemset(d_pDst, 1, vsize);
    assert(err == cudaSuccess);
    NppStatus ret = nppsZero_8u(d_pDst, len);
    assert(ret == NPP_NO_ERROR);
    Npp8u *h_img = new Npp8u[len];
    err = cudaMemcpy(h_img, d_pDst, vsize, cudaMemcpyDeviceToHost);
    // test for zeroing
    for (int i = 0; i < len; i++) assert(!(h_img[i]));

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
