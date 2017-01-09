#########################################################
#                                                       #
#    LQR for Trees                                      #
#    - Arun Venkatraman    <arunvenk@cs.cmu.edu>        #
#                                                       #
#    Created:  Fall 2016                                #
#                                                       #
#########################################################


################## Header  definitions ##################

cmake_minimum_required(VERSION 2.7)
project(lqr_trees)

# Set the default path for built executables to the "bin" directory.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
# Set the default path for built libraries to the "lib" directory.
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(EXTERNALS_DIR ${PROJECT_SOURCE_DIR}/externals)

#########################################################

############### Configuration  Generation ###############

find_package(Eigen3 REQUIRED)
#find_package(Boost COMPONENTS random REQUIRED)
#find_package(OpenMP)

if (CMAKE_BUILD_TYPE)
  string(TOUPPER ${CMAKE_BUILD_TYPE} _build_type)
  set(CMAKE_BUILD_TYPE ${_build_type} CACHE STRING "" FORCE)
endif (CMAKE_BUILD_TYPE)

# Build type must be either DEBUG or RELEASE.
if (CMAKE_BUILD_TYPE STREQUAL "")
  message("No build type specified: forcing to DEBUG.")
  set(CMAKE_BUILD_TYPE "DEBUG" CACHE STRING "" FORCE)
endif (CMAKE_BUILD_TYPE STREQUAL "")

# Linux bitness type.
if (LINUX_BITNESS)
  string(TOUPPER ${LINUX_BITNESS} _linux_bitness)
  set(LINUX_BITNESS ${_linux_bitness} CACHE STRING "" FORCE)
else (LINUX_BITNESS)
  set(LINUX_BITNESS "" CACHE STRING "" FORCE)
endif (LINUX_BITNESS)

# Linux bitness type must be either 32-BIT or 64-BIT.
if (LINUX_BITNESS STREQUAL "")
  EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
  
  if (${ARCHITECTURE} STREQUAL "x86_64")
    message("No linux bitness type specified. Detected 64-bit architecture, forcing to 64-bit bitness compilation.")
    set(LINUX_BITNESS "64-BIT" CACHE STRING "" FORCE)
  elseif (${ARCHITECTURE} STREQUAL "i386")
    message("No linux bitness type specified. Detected 32-bit architecture, forcing to 32-bit bitness compilation.")
    set(LINUX_BITNESS "32-BIT" CACHE STRING "" FORCE)
  else (${ARCHITECTURE} STREQUAL "i386")
    message(FATAL_ERROR "Unknown architecture found. The library cannot be compiled under this architecture yet.")
  endif (${ARCHITECTURE} STREQUAL "x86_64")
endif (LINUX_BITNESS STREQUAL "")

#########################################################

##################### Flag  Setting #####################

set(Extra_CXX_FLAGS "")

#if (OPENMP_FOUND)
#  set(Extra_CXX_FLAGS "${Extra_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -D_USE_OPENMP_")
#  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#ENDIF(OPENMP_FOUND)

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(Extra_CXX_FLAGS "${Extra_CXX_FLAGS} -ggdb")
elseif (CMAKE_BUILD_TYPE STREQUAL "RELEASE")
  set(Extra_CXX_FLAGS "${Extra_CXX_FLAGS} -O3")
else (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  message(FATAL_ERROR "Invalid build type specified: doing nothing.")
endif (CMAKE_BUILD_TYPE STREQUAL "DEBUG")

if (LINUX_BITNESS STREQUAL "32-BIT")
  set(Extra_CXX_FLAGS "${Extra_CXX_FLAGS} -m32")
elseif (LINUX_BITNESS STREQUAL "64-BIT")
  set(Extra_CXX_FLAGS "${Extra_CXX_FLAGS} -m64")
else (LINUX_BITNESS STREQUAL "32-BIT")
  message(FATAL_ERROR "Invalid linux bitness type specified.")
endif (LINUX_BITNESS STREQUAL "32-BIT")

# set to c++11 standard
set(Extra_CXX_FLAGS "${Extra_CXX_FLAGS} -std=c++11")

# Depending on compiler, extra flags may be needed
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang") # matches Apple Clang and linux 
    set(Extra_CXX_FLAGS "${Extra_CXX_FLAGS} -stdlib=libc++")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
endif()

# Turn on all warnings and make them errors.
set(CMAKE_CXX_FLAGS "${Extra_CXX_FLAGS} -Wall -Werror -Wno-error=unused-function -Wno-error=missing-braces")

#########################################################

#################### Code  Compiling ####################

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${EXTERNALS_DIR}/include)
link_directories(${LIBRARY_OUTPUT_PATH})

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)


#################### Add Source Directories ####################
add_subdirectory(src)

#########################################################

################# Software Installation #################

#set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
#
## Headers.
#install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/control_simulators FILES_MATCHING PATTERN "*.h")

# Shared objects.
# install(TARGETS lib/ControlSim LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/control_simulators)

# Binaries.
