cmake_minimum_required(VERSION 3.5)
project(CNNFeaturesDistances)

set(CMAKE_CXX_STANDARD 11)

find_package(HDF5 REQUIRED)
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    link_directories("/usr/lib/x86_64-linux-gnu/hdf5/serial")
    set(HDF5_LIBS hdf5 hdf5_cpp)
endif()

find_package(Boost REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

set(SOURCE_FILES main.cpp Features.cpp Features.h FeaturesHdf5IO.cpp FeaturesHdf5IO.h)
add_executable(CNNFeaturesDistances ${SOURCE_FILES})
target_link_libraries(CNNFeaturesDistances ${HDF5_LIBS})
