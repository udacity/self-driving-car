cmake_minimum_required(VERSION 2.8.3)
project(udacity_place_recognition)
include(ExternalProject)

if(CMAKE_COMPILER_IS_GNUCC)
  ADD_DEFINITIONS ( -std=c++11  )
endif(CMAKE_COMPILER_IS_GNUCC)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  image_transport
  cv_bridge
)

find_package(OpenCV REQUIRED)
find_package(DLib REQUIRED)
find_package(DBoW2 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs image_transport cv_bridge
)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${DLib_INCLUDE_DIRS}
  ${DBoW2_INCLUDE_DIRS}
)

add_executable(udacity_place_recognition_node src/udacity_place_recognition_node.cpp)
add_dependencies(udacity_place_recognition_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(udacity_place_recognition_node
  ${catkin_LIBRARIES} ${DLib_LIBS} ${DBoW2_LIBS} ${OpenCV_LIBS}
)

set(RESOURCES_DIR ${PROJECT_SOURCE_DIR}/resources/)
if(NOT EXISTS ${RESOURCES_DIR})
    ExternalProject_Add(Resources
      PREFIX ${CMAKE_BINARY_DIR}
      URL http://doriangalvez.com/resources/DLoopDetector/resources.tar.gz
      URL_MD5 c001da68ddaceed1de9c16aaac22ed80
      SOURCE_DIR ${RESOURCES_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND "")
    add_custom_target(DownloadResources ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} DEPENDS Resources)
endif(NOT EXISTS ${RESOURCES_DIR})
