cmake_minimum_required (VERSION 3.0)
# set(CMAKE_C_COMPILER, "/usr/bin/gcc-9")
# set(CMAKE_CXX_COMPILER, "/usr/bin/g++-9")
# export CC=/usr/bin/gcc-11 export CXX=/usr/bin/g++-11

project (ACMH)


find_package(CUDA 11.0 REQUIRED) # For Cuda Managed Memory and c++11
find_package(OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)



# torch test ##########################################################################################################################
# set(Torch_DIR /home/yswang/Downloads/software/libtorch-cxx11-abi-shared-with-deps-2.0.0+cu117/libtorch/share/cmake/Torch)
# message("--++ "${CUDNN_LIBRARY_PATH})
# find_package(Torch REQUIRED)
# include_directories(${TORCH_INCLUDE_DIRS})
# find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# include_directories(${Python3_INCLUDE_DIRS})
# find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

# torch test ##########################################################################################################################



# add_subdirectory(extern/pybind11)

# set(CMAKE_BUILD_TYPE Debug) # Debug, Release
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-G -g -O0 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++14 --compiler-options -Wall -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75)
# Debug need -G for cuda -g for cxx

# set(CMAKE_BUILD_TYPE Debug) # Debug Debug need -G for cuda -g for cxx
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-G -g -O0 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++14 --compiler-options -Wall -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75)


set(CMAKE_BUILD_TYPE Release) # Release
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++14 --compiler-options -Wall -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75)



if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++14)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    ACMH
    main.h
    main.cpp

    ACMH.h
    ACMH.cpp
    ACMH.cu

    nccmodule.h
    nccmodule.cu
    # nccmodule.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(ACMH
    ${OpenCV_LIBS}
    )

# torch test ##########################################################################################################################

# target_link_libraries(ACMH ${TORCH_LIBRARIES})
# target_link_libraries(ACMH ${TORCH_PYTHON_LIBRARY})
# target_link_libraries(ACMH ${Python3_LIBRARIES})
# torch test ##########################################################################################################################

# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 20 --slave /usr/bin/g++ g++ /usr/bin/g++-11
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10 --slave /usr/bin/g++ g++ /usr/bin/g++-7
# sudo update-alternatives --config gcc
