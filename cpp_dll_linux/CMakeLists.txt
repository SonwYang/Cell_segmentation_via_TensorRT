cmake_minimum_required(VERSION 3.2)

project(seg)

#SET(LIBDLL_SRC seg_lib.cpp)
#ADD_LIBRARY(seg_lib SHARED ${LIBDLL_SRC})
#ADD_LIBRARY(seg_lib_static STATIC ${LIBDLL_SRC})
#SET_TARGET_PROPERTIES(seg_lib_static PROPERTIES OUTPUT_NAME "seg_lib")

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

find_package(CUDA REQUIRED)

# cuda directory
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/scell/medicalSeg/TensorRT-8.2.0.6/targets/x86_64-linux-gnu/include)
link_directories(/home/scell/medicalSeg/TensorRT-8.2.0.6/targets/x86_64-linux-gnu/lib)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

SET(LIBDLL_SRC seg_lib.cpp)
ADD_LIBRARY(seg_lib SHARED ${LIBDLL_SRC})
target_link_libraries(seg_lib nvinfer cudart)

add_definitions(-O2 -pthread)

# opencv library
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(seg_lib ${OpenCV_LIBS})

