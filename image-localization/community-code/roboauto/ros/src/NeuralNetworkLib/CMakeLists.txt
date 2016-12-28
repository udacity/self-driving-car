cmake_minimum_required(VERSION 3.2)
project(NeuralNetwork CXX)

include(cmake/CPUFeatures)

add_subdirectory(lib/SimpleJSON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Weffc++ -Wshadow -Wstrict-aliasing -ansi -Woverloaded-virtual -Wdelete-non-virtual-dtor -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native -O3")

set(${PROJECT_NAME}_INCLUDE_DIRS
		${PROJECT_SOURCE_DIR}/include
		${SimpleJSON_INCLUDE_DIRS}
		CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

include_directories(${NeuralNetwork_INCLUDE_DIRS} ${SimpleJSON_INCLUDE_DIRS})

catkin_package(
		INCLUDE_DIRS ${${PROJECT_NAME}_INCLUDE_DIRS}
		LIBRARIES NeuralNetwork
)

OPTION(BUILD_SHARED_LIBS "Build also shared library." ON)

if(CPU_AVX_AVAILABLE)
	SET(USE_AVX ON CACHE BOOL "")
	if(CPU_FMA_AVAILABLE)
		SET(USE_FMA ON CACHE BOOL "")
	endif(CPU_FMA_AVAILABLE)
endif(CPU_AVX_AVAILABLE)

if(CPU_SSE3_AVAILABLE)
	SET(USE_SSE ON CACHE BOOL "")

	if(CPU_SSE4.2_AVAILABLE)
		SET(USE_SSE2 OFF CACHE BOOL "")
	endif(CPU_SSE4.2_AVAILABLE)

endif(CPU_SSE3_AVAILABLE)

OPTION(USE_SSE "If SSE instruction set should be used." OFF)
OPTION(USE_SSE2 "If SSE 2 instruction set should be used." ON)
OPTION(USE_AVX "If AVX instruction set should be used." OFF)
OPTION(USE_FMA "If FMA instruction set should be used." OFF)

OPTION(ENABLE_TESTS "enables tests" ON)

if(USE_AVX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -DUSE_AVX")
	if(USE_FMA)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma -DUSE_FMA")
	endif(USE_FMA)

endif(USE_AVX)

if(USE_SSE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -DUSE_SSE")
	if(USE_SSE2)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SSE2")
	endif(USE_SSE2)
endif(USE_SSE)

set (LIBRARY_SOURCES
	src/NeuralNetwork/Learning/BatchPropagation.cpp
	src/NeuralNetwork/Learning/BackPropagation.cpp
	src/NeuralNetwork/Learning/QuickPropagation.cpp
	src/NeuralNetwork/Learning/PerceptronLearning.cpp
	src/NeuralNetwork/Learning/RProp.cpp
	src/NeuralNetwork/Learning/iRPropPlus.cpp

	src/NeuralNetwork/ConstructiveAlgorithms/CascadeCorrelation.cpp
	src/NeuralNetwork/ConstructiveAlgorithms/Cascade2.cpp

	src/NeuralNetwork/BasisFunction/Linear.cpp

	src/NeuralNetwork/FeedForward/Layer.cpp
	src/NeuralNetwork/FeedForward/Network.cpp
	src/NeuralNetwork/Recurrent/Network.cpp

	src/NeuralNetwork/Neuron.cpp
	src/NeuralNetwork/IMPL.cpp
)

add_library(NeuralNetwork STATIC ${LIBRARY_SOURCES})

link_libraries(NeuralNetwork pthread)

IF(BUILD_SHARED_LIBS)
	add_library(NeuralNetworkShared SHARED ${LIBRARY_SOURCES})
	set_target_properties(NeuralNetworkShared PROPERTIES OUTPUT_NAME NeuralNetwork)
ENDIF(BUILD_SHARED_LIBS)