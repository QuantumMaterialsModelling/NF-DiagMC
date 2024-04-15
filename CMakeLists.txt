#************************* #
#---PROJECT DESCRIPTION--- #
#************************* #
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
  project(NFDMC
  VERSION 0.0.1
  DESCRIPTION "Multi purpouse Diagrammatic Monte Carlo code"
  LANGUAGES CXX)
endif()

#****************** #
#---DEPENDENCIES--- #
#****************** #
find_package(Boost REQUIRED)
find_package(Torch)

#*********************** #
#---GLOBAL PROPERTIES--- #
#*********************** #
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#***************************** #
#---ENTERING SUBDIRECTORIES--- #
#***************************** #
add_subdirectory(src)
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} AND ${Torch_FOUND})
  #*********************************** #
  #---COMPILING HOLSTEIN IF IS MAIN--- #
  #*********************************** #
  add_subdirectory(Holstein)
endif()
