# TODO: make this an actual find module...allow VERSION, QUIET, etc

# TODO: don't use find_package(CUDA)?  But the thread specifically states that
#       we should *NOT* require that enable_language(CUDA) has been done,
#       meaning that e.g. CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES may not exist?
#
#       Solution?
#       include(CheckLanguage)
#       check_language(CUDA)
#       if (NOT CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
#       --> then we cannot win?
find_package(CUDA REQUIRED)

# Populate the list of default locations to search for the CUDA libraries.
# TODO: allow user bypass of this?
list(APPEND CUDALibs_HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
list(APPEND CUDALibs_HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib")
list(APPEND CUDALibs_HINTS "${CUDA_TOOLKIT_ROOT_DIR}")

function(find_and_add_cuda_import_lib lib_name)
  string(TOUPPER ${lib_name} LIB_NAME)
  find_library(CUDA_${LIB_NAME} ${lib_name} HINTS ${CUDALibs_HINTS})
  if (NOT CUDA_${LIB_NAME} STREQUAL CUDA_${LIB_NAME}-NOTFOUND)
    add_library(CUDA::${lib_name} IMPORTED INTERFACE)
    set_target_properties(CUDA::${lib_name}
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
          "${CUDA_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES
          "${CUDA_${LIB_NAME}}"
    )
  endif()
endfunction()

# TODO: how to make sure `dependency` can actually be used
# TODO: if dependency cannot be used, is it possible to
#       delete CUDA::${lib_name}?
function(add_cuda_link_dependency lib_name dependency)
  set_property(
    TARGET CUDA::${lib_name}
    APPEND
    PROPERTY
      INTERFACE_LINK_LIBRARIES ${dependency}
  )
endfunction()

# Find the main CUDA runtime dynamic and static libraries.
# These are a hard dependency for all other libraries, and
# must be found.
# TODO: right way to error out?
find_and_add_cuda_import_lib(cudart)
find_and_add_cuda_import_lib(cudart_static)

# TODO: what about windows???
if (UNIX)
  foreach (lib dl pthread rt)
    add_cuda_link_dependency(cudart_static ${lib})
  endforeach()
endif()

# TODO: NPPI and various libs, need examples to test with:
#       https://docs.nvidia.com/cuda/npp/index.html
#
#       * nppicc  color conversion and sampling functions in nppi_color_conversion.h
#       * nppicom JPEG compression and decompression functions in nppi_compression_functions.h
#       * nppidei data exchange and initialization functions in nppi_data_exchange_and_initialization.h
#       * nppif   filtering and computer vision functions in nppi_filter_functions.h
#       * nppig   geometry transformation functions found in nppi_geometry_transforms.h
#       * nppim   morphological operation functions found in nppi_morphological_operations.h
#       * nppist  statistics and linear transform in nppi_statistics_functions.h and nppi_linear_transforms.h
#       * nppisu  memory support functions in nppi_support_functions.h
#       * nppitc  threshold and compare operation functions in nppi_threshold_and_compare_operations.h
find_and_add_cuda_import_lib(nppc)
find_and_add_cuda_import_lib(nppc_static)

foreach (cuda_lib nppial)
  # Find the NPP library.
  find_and_add_cuda_import_lib(${cuda_lib})
  find_and_add_cuda_import_lib(${cuda_lib}_static)

  # Dynamic link dependencies.
  # add_cuda_link_dependency(${cuda_lib} CUDA::nppc)
  add_cuda_link_dependency(${cuda_lib} CUDA::cudart)

  # Static link dependencies.
  add_cuda_link_dependency(${cuda_lib}_static CUDA::nppc_static)
  add_cuda_link_dependency(${cuda_lib}_static CUDA::cudart_static)
  add_cuda_link_dependency(${cuda_lib}_static CUDA::culibos)
endforeach()

# TODO: nvBLAS and example.  Depends on cuBLAS, but not sure how it works.
#       Testing executable may need to find_package(BLAS)?  It seems like the idea is
#       you write a standard BLAS level 3 operation, and at link time nvBLAS will take
#       over somehow?

# TODO: (nppi* nvblas)
foreach (cuda_lib cublas cufft cufftw curand cusolver cusparse nvgraph)
  # find the dynamic library
  find_and_add_cuda_import_lib(${cuda_lib})
  add_cuda_link_dependency(${cuda_lib} CUDA::cudart)

  # TODO: if UNIX and VERSION >= 6.5
  # find the static library
  find_and_add_cuda_import_lib(${cuda_lib}_static)
  add_cuda_link_dependency(${cuda_lib}_static CUDA::cudart_static)
endforeach()

# NVRTC (Runtime Compilation) is a shared library only.
# TODO: nvrtc needs -lcuda (*NOT* cudart), but -lcuda (at least on this system)
#       is going to point to /lib64/libcuda.so.
#
#       Since this is not in the HINTS paths searched above, what is the right
#       way to create the CUDA::cuda target?
find_and_add_cuda_import_lib(nvrtc)
add_cuda_link_dependency(nvrtc cuda)

# NVTX is a shared library only.
# TODO: is this even useful outside of NSight Eclipse?
find_and_add_cuda_import_lib(nvToolsExt)
add_cuda_link_dependency(nvToolsExt CUDA::cudart)

# cuLIBOS is a static only library, see
#
# https://devblogs.nvidia.com/10-ways-cuda-6-5-improves-performance-productivity
#
# > Static CUDA Libraries
# > CUDA 6.5 (on Linux and Mac OS) now includes static library versions of the
# > cuBLAS, cuSPARSE, cuFFT, cuRAND, and NPP libraries. This can reduce the
# > number of dynamic library dependencies you need to include with your
# > deployed applications. These new static libraries depend on a common thread
# > abstraction layer library cuLIBOS (libculibos.a) distributed as part of the
# > CUDA toolkit.
find_and_add_cuda_import_lib(culibos)
# foreach (cuda_lib cublas cusparse cufft) # curand npp
foreach (cuda_lib cublas cufft cusparse curand)# npp
  add_cuda_link_dependency(${cuda_lib}_static CUDA::culibos)
endforeach()

# cuSOLVER depends on cuBLAS and cuSPARSE
# NOTE: nvGRAPH relies on this, make sure it happens before nvGRAPH dependencies.
foreach (dep cublas cusparse)
  add_cuda_link_dependency(cusolver CUDA::${dep})
  add_cuda_link_dependency(cusolver_static CUDA::${dep}_static)
endforeach()

# nvGRAPH depends on cuBLAS, cuRAND, cuSPARSE, and cuSOLVER.
# NOTE: rely on link dependencies of cuSOLVER, this must happen after cusolver target.
foreach (dep cusolver curand)
  add_cuda_link_dependency(nvgraph CUDA::${dep})
  add_cuda_link_dependency(nvgraph_static CUDA::${dep}_static)
endforeach()

# TODO: mysterious extra static libraries...what are they for?
find_and_add_cuda_import_lib(cudadevrt)
find_and_add_cuda_import_lib(cublas_device)

# TODO: VERSION 9.2, search libcufft_static_nocallback.a
#       https://docs.nvidia.com/cuda/cufft/index.html#oned-complex-to-complex-transforms

# Do not expose these functions externally.
unset(find_and_add_cuda_import_lib)
unset(add_cuda_link_dependency)
