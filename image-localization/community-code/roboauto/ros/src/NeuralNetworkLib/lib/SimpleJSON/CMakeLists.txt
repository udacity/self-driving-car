cmake_minimum_required(VERSION 3.2)
project(SimpleJSON)

include_directories(include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g")
OPTION(ENABLE_TESTS "Build tests." OFF)

set(SimpleJSON_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/include
	CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

