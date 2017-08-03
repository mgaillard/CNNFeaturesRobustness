cmake_minimum_required(VERSION 3.5)
project(CNNFeaturesBenchmark)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

if(WITH_NATIVE_ARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(HDF5 REQUIRED)
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    link_directories("/usr/lib/x86_64-linux-gnu/hdf5/serial")
    set(HDF5_LIBS hdf5 hdf5_cpp)
endif()

set(Boost_USE_STATIC_LIBS        ON) # only find static libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

set(SOURCE_FILES src/main.cpp src/FeaturesHdf5IO.cpp src/FeaturesHdf5IO.h src/CnnFeatures.cpp src/CnnFeatures.h src/Benchmark.h src/BenchmarkStats.cpp src/BenchmarkStats.h src/CnnCode.h src/CnnCode.cpp src/Index.h)
add_executable(CNNFeaturesBenchmark ${SOURCE_FILES})
target_link_libraries(CNNFeaturesBenchmark ${HDF5_LIBS})
target_link_libraries(CNNFeaturesBenchmark ${Boost_LIBRARIES})

set(SOURCE_FILES_TEST test/test.cpp test/catch.hpp src/CnnFeatures.cpp src/CnnFeatures.h src/CnnCode.h src/CnnCode.cpp src/Index.h src/Benchmark.h src/BenchmarkStats.cpp src/BenchmarkStats.h)
add_executable(CNNFeaturesBenchmarkTest ${SOURCE_FILES_TEST})