cmake_minimum_required(VERSION 3.9...3.12 FATAL_ERROR)
project(cmake_cuda_libs_tests LANGUAGES CXX)

# eventually this gets added to an MR to CMake proper...
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

find_package(CUDALibs REQUIRED)

# adds two executables from same source tests/${cuda_lib}
# for testing both dynamic and static linkage
function(make_both_tests cuda_lib)
  # dynamic linkage test
  add_executable(test-${cuda_lib} tests/${cuda_lib}.cpp)
  target_link_libraries(test-${cuda_lib} CUDA::${cuda_lib})

  # static linkage test
  add_executable(test-${cuda_lib}-static tests/${cuda_lib}.cpp)
  target_link_libraries(test-${cuda_lib}-static CUDA::${cuda_lib}_static)
endfunction()

make_both_tests(cudart)
make_both_tests(cublas)

# TODO: cufft_static is BROKEN, maybe impossible to link without `nvlink`?
# make_both_tests(cufft)
# TODO: difference between cufft and cufftw?

make_both_tests(curand)
make_both_tests(cusolver)
make_both_tests(cusparse)

# NPP component libraries.
make_both_tests(nppial)
make_both_tests(nppicc)
make_both_tests(nppicom)
make_both_tests(nppidei)
make_both_tests(nppif)
make_both_tests(nppig)
make_both_tests(nppim)
make_both_tests(nppist)
make_both_tests(nppisu)

# TODO: nvblas

make_both_tests(nvgraph)

# NVRTC (Runtime Compilation) is a shared library only.
add_executable(test-nvrtc tests/nvrtc.cpp)
target_link_libraries(test-nvrtc CUDA::nvrtc)

# NVTX is only meaningful in the context of compiling CUDA code.
# The sample tests/nvtx comes from the parallel for all blog post
# https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
#
# The below CMake code as an adaptation of the provided Makefile:
# https://github.com/parallel-forall/code-samples/blob/master/posts/nvtx/Makefile
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)

  add_executable(test-manual-nvtx tests/nvtx/manual_nvtx.cu)
  target_compile_definitions(test-manual-nvtx PUBLIC -DUSE_NVTX)
  target_link_libraries(test-manual-nvtx CUDA::nvToolsExt)

  add_executable(test-compiler-inst-nvtx tests/nvtx/compiler_inst_nvtx.cu tests/nvtx/inst_nvtx.cpp)
  set_property(TARGET test-compiler-inst-nvtx PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_options(test-compiler-inst-nvtx PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-finstrument-functions>)
  target_link_libraries(test-compiler-inst-nvtx CUDA::nvToolsExt)

  # is -export-dynamic available on windows?
  if (UNIX)
    foreach (tgt test-manual-nvtx test-compiler-inst-nvtx)
      target_compile_options(${tgt} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-export-dynamic>)
      target_compile_options(${tgt} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-export-dynamic>)
    endforeach()
  endif()

  add_custom_target(nvtx ALL
    # First command
    nvprof -f -o compiler_inst_nvtx.nvvp $<TARGET_FILE:test-compiler-inst-nvtx>
    # Second command
    COMMAND nvprof -f -o manual_nvtx.nvvp $<TARGET_FILE:test-manual-nvtx>
    DEPENDS test-manual-nvtx test-compiler-inst-nvtx
  )
endif()
