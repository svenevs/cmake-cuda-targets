////////////////////////////////////////////////////////////////////////////////
// I MAKE NO CLAIMS TO THIS FILE, @txbob (the great) KINDLY SUPPLIED THIS:    //
// https://devtalk.nvidia.com/default/topic/1037482/gpu-accelerated-libraries/help-me-help-you-with-modern-cmake-and-cuda-mwe-for-npp/post/5271059/#5271059
////////////////////////////////////////////////////////////////////////////////
#include <nppi_compression_functions.h>
#include <assert.h>
#include <iostream>

int main(void) {
    /**
     * Returns the length of the NppiDecodeHuffmanSpec structure.
     */
    int pSize = 0;
    NppStatus ret = nppiDecodeHuffmanSpecGetBufSize_JPEG(&pSize);
    assert(ret == NPP_NO_ERROR);
    assert(pSize > 0);

    std::cout << "Test ran successfully!" << std::endl;

    return 0;
}
