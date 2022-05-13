cmake_minimum_required(VERSION 3.2)

project(seg)
set(OpenCV_DIR "D:\\MyWorkSpace\\Lib\\opencv\\build")  #2
set(TRT_DIR "D:\\MyWorkSpace\\Lib\\TensorRT-8.2.1.8")  #2 
set(OpenCV_INCLUDE_DIRS "D:\\MyWorkSpace\\Lib\\opencv\\build\\include")   #3
set(OpenCV_LIBS "D:\\MyWorkSpace\\Lib\\opencv\\build\\x64\\vc14\\lib\\opencv_world340.lib")   #4

SET(LIBDLL_SRC seg_lib.cpp)
ADD_LIBRARY(seg_lib SHARED ${LIBDLL_SRC})
ADD_LIBRARY(seg_lib_static STATIC ${LIBDLL_SRC})
SET_TARGET_PROPERTIES(seg_lib_static PROPERTIES OUTPUT_NAME "seg_lib")

add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
target_link_libraries(seg_lib nvinfer cudart)

include_directories(${TRT_DIR}\\include)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS}\\opencv2)  #6

##### find package(opencv)
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

target_link_libraries(seg_lib ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${TRT_DIR}\\lib)           #7
link_directories(${OpenCV_DIR}\\x64\\vc14\\lib)                         #8

add_definitions(-O2 -pthread)

