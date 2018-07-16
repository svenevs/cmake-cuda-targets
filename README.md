# cmake-cuda-targets

This repository is being used to implement `FindCUDALibs.cmake` for making
the NVIDIA CUDA Toolkit libraries such as `cuBLAS` or `cuSOLVER` available in
CMake.  This work is intended to resolve issue
[cmake/#17816](https://gitlab.kitware.com/cmake/cmake/issues/17816).

> **WARNING**: this is very much a work in progress, not ready for production
> use.  You are advised to resist using the `FindCUDALibs.cmake` found here,
> it will become a part of CMake proper when it has been fleshed out / tested
> by more users.

The intent of `FindCUDALibs.cmake` is to enable convenient usage of the NVIDIA
libraries in pure C/C++, as the majority of these APIs do not require writing
native CUDA code.  As such, all of the examples (except for NVTX) are just
samples grabbed from the NVIDIA documentation for each respective library.

The listing of libraries was ascertained by listing the contents of my local
Linux CUDA installation under `/usr/local/cuda/lib64`.  Some libraries are not
(currently) included, as I do not know what they do...see
[Miscellaneous Un-handled Libraries](#miscellaneous-un-handled-libraries).

When linking against a given `CUDA::xxx` library vs a `CUDA::xxx_static`
library, any dependencies encoded will cascade where dynamic vs static libraries
are concerned.  For example, `cuSOLVER` depends on both `cuBLAS` and `cuSPARSE`.
So

- Linking against `CUDA::cusolver` links dynamically against `CUDA::cublas` and
  `CUDA::cusparse`.
- Linking against `CUDA::cusolver_static` links statically against
  `CUDA::cublas_static` and `CUDA::cusparse_static`.

When choosing static linkage, the `CUDA::culibos` static library will also
automatically be included in your linking dependencies for the libraries that
need this.  See the [cuLIBOS](#culibos) section.

**TODO**: there are **many** `TODO` left, both in this README as well as in the
implementation ([`FindCUDALibs.cmake`](cmake/Modules/FindCUDALibs.cmake)).

**Contents**

- [CUDA Runtime Libraries](#cuda-runtime-libraries)
- [cuBLAS](#cublas)
- [cuFFT](#cufft)
- [cuFFTW](#cufftw)
- [cuLIBOS](#culibos)
- [cuRAND](#curand)
- [cuSOLVER](#cusolver)
- [cuSPARSE](#cusparse)
- [NPP](#npp)
- [nvBLAS](#nvblas)
- [nvGRAPH](#nvgraph)
- [nvToolsExt](#nvtoolsext)
- [Miscellaneous Un-handled Libraries](#miscellaneous-un-handled-libraries)

# CUDA Runtime Libraries

The CUDA Runtime libraries (cudart) are what most applications will typically
need to link against to make any calls such as `cudaMalloc` and `cudaFree`.
They are an explicit dependency of almost every library.

**Targets Created**:

- `CUDA::cudart`
    - `libcudart.so@`
- `CUDA::cudart_static`
    - `libcudart_static.a`

# cuBLAS

The [`cuBLAS` library](https://docs.nvidia.com/cuda/cublas/index.html).

**Targets Created**:

- `CUDA::cublas`
    - `libcublas.so@`

- `CUDA::cublas_static`
    - `libcublas_static.a`

**Testing Program**: [`tests/cublas.cpp`](tests/cublas.cpp)

**TODO**: what is `libcublas_device.a` for?

# cuFFT

The [`cuFFT` library](https://docs.nvidia.com/cuda/cufft/index.html).

**Targets Created**:

- `CUDA::cufft`
    - `libcufft.so@`
- `CUDA::cufft_static`
    - `libcufft_static.a`

**Testing Program** [`tests/cufft.cpp`](tests/cufft.cpp)

**TODO**: `cufft_static` test program is not able to link which is why it is
commented out in `CMakeLists.txt`.

# cuFFTW

It is unclear what the difference is between `cuFFT` and `cuFFTW` to the author
of this document.

**TODO**: after fixing `cufft_static`, create `CUDA::cufftw` and
`CUDA::cufftw_static` targets.

- `libcufftw.so@`
- `libcufftw_static.a`

# cuLIBOS

The cuLIBOS library is a backend thread abstraction layer library which is
static only.  The `CUDA::cublas_static`, `CUDA::cusparse_static`,
`CUDA::cufft_static`, `CUDA::curand_static`, and (when implemented) NPP
libraries all automatically have this dependency linked.  Search for the phrase
`Static CUDA Libraries` on
[this blog post](https://devblogs.nvidia.com/10-ways-cuda-6-5-improves-performance-productivity/).

**Target Created**:

- `CUDA::culibos`
    - `libculibos.a`

**Test Program**: not applicable, e.g., `cublas` static executable tests this.

**Note**: direct usage of this target by consumers should not be necessary.

# cuRAND

The [`cuRAND` library](https://docs.nvidia.com/cuda/curand/index.html).

**Targets Created**:

- `CUDA::curand`
    - `libcurand.so@`
- `CUDA::curand_static`
    - `libcurand_static.a`

**Testing Program**: [`tests/curand.cpp`](tests/curand.cpp)

# cuSOLVER

The [`cuSOLVER` library](https://docs.nvidia.com/cuda/cusolver/index.html).

**Targets Created**:

- `CUDA::cusolver`
    - `libcusolver.so@`
- `CUDA::cusolver_static`
    - `libcusolver_static.a`

**Testing Program**: [`tests/cusolver.cpp`](tests/cusolver.cpp)

# cuSPARSE

The [`cuSPARSE` library](https://docs.nvidia.com/cuda/cusparse/index.html).

**Targets Created**:

- `CUDA::cusparse`
    - `libcusparse.so@`
- `CUDA::cusparse_static`
    - `libcusparse_static.a`

# NPP

The [`NPP` libraries](https://docs.nvidia.com/cuda/npp/index.html).

**Targets Created**:

- `nppc`: **TODO**: descrption ???
    - `CUDA::nppc`
        - `libnppc.so@`
    - `CUDA::nppc_static`
        - `libnppc_static.a`

    **Testing Program**: **TODO** ??? it's statically linked in many
- `nppial`: *arithmetic and logical operation functions in `nppi_arithmetic_and_logical_operations.h`*
    - `CUDA::nppial`
        - `libnppial.so@`
    - `CUDA::nppial_static`
        - `libnppial_static.a`

    **Testing Program**: [`tests/nppial.cpp`](tests/nppial.cpp)
- `nppicc`: *color conversion and sampling functions in `nppi_color_conversion.h`*
    - `CUDA::nppicc`
        - `libnppicc.so@`
    - `CUDA::nppicc_static`
        - `libnppicc_static.a`

    **Testing Program**: [`tests/nppicc.cpp`](tests/nppicc.cpp)
- `nppicom`: *JPEG compression and decompression functions in `nppi_compression_functions.h`*
    - `CUDA::nppicom`
        - `libnppicom.so@`
    - `CUDA::nppicom_static`
        - `libnppicom_static.a`

    **Testing Program**: [`tests/nppicom.cpp`](tests/nppicom.cpp)
- `nppidei`: *data exchange and initialization functions in `nppi_data_exchange_and_initialization.h`*
    - `CUDA::nppidei`
        - `libnppidei.so@`
    - `CUDA::nppidei_static`
        - `libnppidei_static.a`

    **Testing Program**: [`tests/nppidei.cpp`](tests/nppidei.cpp)
- `nppif`: *filtering and computer vision functions in `nppi_filter_functions.h`*
    - `CUDA::nppif`
        - `libnppif.so@`
    - `CUDA::nppif_static`
        - `libnppif_static.a`

    **Testing Program**: [`tests/nppif.cpp`](tests/nppif.cpp)
- `nppig`: *geometry transformation functions found in `nppi_geometry_transforms.h`*
    - `CUDA::nppig`
        - `libnppig.so@`
    - `CUDA::nppig_static`
        - `libnppig_static.a`

    **Testing Program**: [`tests/nppig.cpp`](tests/nppig.cpp)
- `nppim`: *morphological operation functions found in `nppi_morphological_operations.h`*
    - `CUDA::nppim`
        - `libnppim.so@`
    - `CUDA::nppim_static`
        - `libnppim_static.a`

    **Testing Program**: [`tests/nppim.cpp`](tests/nppim.cpp)
- `nppist`: *statistics and linear transform in `nppi_statistics_functions.h` and `nppi_linear_transforms.h`*
    - `CUDA::nppist`
        - `libnppist.so@`
    - `CUDA::nppist_static`
        - `libnppist_static.a`

    **Testing Program**: [`tests/nppist.cpp`](tests/nppist.cpp)
- `nppisu`: *memory support functions in `nppi_support_functions.h`*
    - `CUDA::nppisu`
        - `libnppisu.so@`
    - `CUDA::nppisu_static`
        - `libnppisu_static.a`

    **Testing Program**: [`tests/nppisu.cpp`](tests/nppisu.cpp)
- `nppitc`: *threshold and compare operation functions in `nppi_threshold_and_compare_operations.h`*
    - `CUDA::nppitc`
        - `libnppitc.so@`
    - `CUDA::nppitc_static`
        - `libnppitc_static.a`

    **Testing Program**: [`tests/nppitc.cpp`](tests/nppitc.cpp)
- `npps`: **TODO**: descrption ???
    - `CUDA::npps`
        - `libnpps.so@`
    - `CUDA::npps_static`
        - `libnpps_static.a`

    **Testing Program**: [`tests/npps.cpp`](tests/npps.cpp)

# nvBLAS

The [`nvBLAS` libraries](https://docs.nvidia.com/cuda/nvblas/index.html).

**TODO**: neither of these targets are currently created, as it is unclear how
to test.  It seems I will need to `find_package(BLAS)` and call some level 3
operations (e.g., stick with `gemm` for simplicity?), but it is unclear how
to both enforce and verify GPU dispatch.

- `CUDA::nvblas`
    - `libnvblas.so@`

No static (which makes sense).


# nvGRAPH

The [`nvGRAPH` library](https://docs.nvidia.com/cuda/nvgraph/index.html).

**Targets Created**:

- `CUDA::nvgraph`
    - `libnvgraph.so@`
- `CUDA::nvgraph_static`
    - `libnvgraph_static.a`

**Testing Program**: [`tests/nvgraph.cpp`](tests/nvgraph.cpp)

# nvRTC

The [`nvRTC` (Runtime Compilation) library](https://docs.nvidia.com/cuda/nvrtc/index.html).

This is a shared library only.

**Targets Created**:

- `CUDA::nvrtc`
    - `libnvrtc.so@`


**Testing Program**: [`tests/nvrtc.cpp`](tests/nvrtc.cpp)

**TODO**: the `libnvrtc-builtins.so@` shared library currently does not have a
target created, should it?  How do I test this specifically?

# nvToolsExt

The [NVIDIA Tools Extension](http://developer.download.nvidia.com/NsightVisualStudio/2.2/Documentation/UserGuide/HTML/Content/NVIDIA_Tools_Extension_Library_NVTX.htm), see also
[this blog post](https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/).

This is a shared library only.

**Targets Created**:

- `CUDA::nvToolsExt`
    - `libnvToolsExt.so@`

**Testing Programs**: this tooling library only makes sense with some native
CUDA code to actually instrument.  Two applications are created, and by default
the `all` target will also run `nvprof` on these to generate two `.nvvp` files.

# Miscellaneous Un-handled Libraries

- `libaccinj64.so@`
- `libcublas_device.a`
- `libcudadevrt.a`
- `libcuinj64.so@`

Libraries pertaining to OpenCL that should not be included:

- `libOpenCL.so@`
