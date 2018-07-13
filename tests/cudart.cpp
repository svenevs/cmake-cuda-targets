// just run a simple memcpy to device and back to make sure cudart is working
#include <cuda_runtime.h>
#include <iostream>

#define N 100

int main(void) {
    // initialize h_memory_src as 0..N, and copy to device.
    // then copy back into a zerod out array and make sure
    // everything matches up
    float *h_memory_src = (float *)malloc(N * sizeof(float));
    float *h_memory_dst = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_memory_src[i] = (float) i;
        h_memory_dst[i] = 0.0f;
    }

    // allocate device buffer to copy h_memory_src to device
    float *d_memory = NULL;
    cudaMalloc((void **)&d_memory, N * sizeof(float));
    cudaMemcpy(
        d_memory,              // destination
        h_memory_src,          // source
        N * sizeof(float),     // count (bytes)
        cudaMemcpyHostToDevice // transfer kind
    );

    cudaDeviceSynchronize();// make sure transfer H->D is complete

    // copy back to h_memory_dst (which is all zeros right now)
    cudaMemcpy(
        h_memory_dst,          // destination
        d_memory,              // source
        N * sizeof(float),     // count (bytes)
        cudaMemcpyDeviceToHost // transfer kind
    );

    cudaDeviceSynchronize();// make sure transfer D->H is complete

    int num_valid = 0;
    for (int i = 0; i < N; ++i) {
        if (h_memory_src[i] == h_memory_dst[i])
            ++num_valid;
    }

    if (num_valid == N)
        std::cout << "All " << N << " floats are the same!" << std::endl;
    else
        std::cout << num_valid << " / " << N << " floats are the same..." << std::endl;

    free(h_memory_src);
    free(h_memory_dst);
    cudaFree(d_memory);

    return 0;
}
