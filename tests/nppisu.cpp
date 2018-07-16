////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271078/#5271078
////////////////////////////////////////////////////////////////////////////////
#include <nppi_support_functions.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * One-channel 8-bit unsigned allocation
     */
    const int simgrows = 32;
    const int simgcols = 32;
    int pitch;
    Npp8u *d_ptr = NULL;
    d_ptr =  nppiMalloc_8u_C1 (simgcols, simgrows, &pitch);
    assert(d_ptr != NULL);

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
