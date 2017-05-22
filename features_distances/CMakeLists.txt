cmake_minimum_required(VERSION 3.5)
project(CNNFeaturesDistances)

set(CMAKE_CXX_STANDARD 11)

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

set(SOURCE_FILES main.cpp FeaturesHdf5IO.cpp FeaturesHdf5IO.h CnnFeatures.cpp CnnFeatures.h)
add_executable(CNNFeaturesDistances ${SOURCE_FILES})
target_link_libraries(CNNFeaturesDistances ${HDF5_LIBS})
target_link_libraries(CNNFeaturesDistances ${Boost_LIBRARIES})
