cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM3)

# Build type (default to Release)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Debug postfix
set(CMAKE_DEBUG_POSTFIX "_d")

# Check C++11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
cmake_policy(SET CMP0057 NEW) # enable IN_LIST operator
if(cxx_std_11 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  add_definitions(-DCOMPILEDWITHC11)
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Compiler specific options for gcc
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

  # suppress warnings about deprecations (many in Eigen)
  add_compile_options("-Wno-deprecated-declarations")

  message(STATUS "ORB_SLAM3 is using compiler flag -march=native. Make sure all relevant libraries are consistent with this.")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()
# Compiler specific options for MSVC
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /bigobj")
  # use the static CRT if other libraries are compiled with /MT
  # (CMake sets /MD by default, but for example Pangolin uses /MT by default)
  #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
if ("${CMAKE_GENERATOR}" MATCHES "Visual Studio")
  # we don't support the Debug and Release subdirs
  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${config}" config_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  endforeach ()
endif ()



# Target definition
set(TARGET_ORB_SLAM3 ORB_SLAM3)
set(ORB_SLAM3_LIB_TYPE SHARED)
IF(WIN32)
  # uncomment only one of the lines below:
  # because there is no DLL API in the code,
  # we either have to build STATIC libraries,
  set(ORB_SLAM3_LIB_TYPE STATIC)
  # or we have to export all symbols from all SHARED libraries
  #set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  # Note that the ideal solution would be to add an export API to the relevant classes.
ENDIF(WIN32)


set(ORB_SLAM3_SOURCES
  src/System.cc
  src/Tracking.cc
  src/LocalMapping.cc
  src/LoopClosing.cc
  src/ORBextractor.cc
  src/ORBmatcher.cc
  src/FrameDrawer.cc
  src/Converter.cc
  src/MapPoint.cc
  src/KeyFrame.cc
  src/Atlas.cc
  src/Map.cc
  src/MapDrawer.cc
  src/Optimizer.cc
  src/Frame.cc
  src/KeyFrameDatabase.cc
  src/Sim3Solver.cc
  src/Viewer.cc
  src/ImuTypes.cc
  src/G2oTypes.cc
  src/CameraModels/Pinhole.cpp
  src/CameraModels/KannalaBrandt8.cpp
  src/OptimizableTypes.cpp
  src/MLPnPsolver.cpp
  src/GeometricTools.cc
  src/TwoViewReconstruction.cc
  src/Config.cc
  src/Settings.cc
)

set(ORB_SLAM3_HEADERS
  include/System.h
  include/Tracking.h
  include/LocalMapping.h
  include/LoopClosing.h
  include/ORBextractor.h
  include/ORBmatcher.h
  include/FrameDrawer.h
  include/Converter.h
  include/MapPoint.h
  include/KeyFrame.h
  include/Atlas.h
  include/Map.h
  include/MapDrawer.h
  include/Optimizer.h
  include/Frame.h
  include/KeyFrameDatabase.h
  include/Sim3Solver.h
  include/Viewer.h
  include/ImuTypes.h
  include/G2oTypes.h
  include/CameraModels/GeometricCamera.h
  include/CameraModels/Pinhole.h
  include/CameraModels/KannalaBrandt8.h
  include/OptimizableTypes.h
  include/MLPnPsolver.h
  include/GeometricTools.h
  include/TwoViewReconstruction.h
  include/SerializationUtils.h
  include/Config.h
  include/Settings.h
)

add_library(${TARGET_ORB_SLAM3} ${ORB_SLAM3_LIB_TYPE}
  ${ORB_SLAM3_HEADERS} ${ORB_SLAM3_SOURCES}
)

target_include_directories(${TARGET_ORB_SLAM3} PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/CameraModels
)

option(ORB_SLAM3_USE_OWN_DBoW2 "Use DBoW2 from Thirdparty folder" TRUE)
option(ORB_SLAM3_USE_OWN_g2o "Use g2o from Thirdparty folder" TRUE)
option(ORB_SLAM3_USE_OWN_hashlibpp "Use hashlibpp from Thirdparty folder" TRUE)
option(ORB_SLAM3_USE_OWN_Sophus "Use Sophus from Thirdparty folder" TRUE)
option(ORB_SLAM3_BUILD_EXAMPLES "Build examples" TRUE)
option(ORB_SLAM3_BUILD_EXAMPLES_OLD "Build old examples" TRUE)

# Dependencies

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# DBoW2
if(ORB_SLAM3_USE_OWN_DBoW2)
  add_subdirectory(Thirdparty/DBoW2)
  target_include_directories(${TARGET_ORB_SLAM3} PUBLIC Thirdparty/DBoW2)
else()
  find_package(DBoW2 REQUIRED)
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC DBoW2)

# G2O
if(ORB_SLAM3_USE_OWN_g2o)
  add_subdirectory(Thirdparty/g2o)
  target_include_directories(${TARGET_ORB_SLAM3} PUBLIC Thirdparty/g2o)
else()
  find_package(g2o REQUIRED)
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC g2o)

# Sophus
if(ORB_SLAM3_USE_OWN_Sophus)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(BUILD_TESTS OFF)
  set(BUILD_EXAMPLES OFF)
  add_subdirectory(Thirdparty/Sophus)
  # (include public because used in OS3 headers)
  target_include_directories(${TARGET_ORB_SLAM3} PUBLIC Thirdparty/Sophus)
else()
  find_package(Sophus REQUIRED)
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC sophus)

# hashlibpp
if(ORB_SLAM3_USE_OWN_hashlibpp)
  add_subdirectory(Thirdparty/hashlibpp)
else()
  find_package(hashlibpp REQUIRED)
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PRIVATE hashlibpp)
target_compile_definitions(${TARGET_ORB_SLAM3} PRIVATE USE_HASHLIBPP)

# OpenSSL is used only in System.cc for MD5 hash calculation
# but because OpenSSL is not trivial to compile on windows, we can use hashlibpp instead.
#set(OPENSSL_USE_STATIC_LIBS TRUE)
#find_package(OpenSSL) # for crypto library
#if(OpenSSL_FOUND)
#  target_include_directories(${TARGET_ORB_SLAM3} ${OPENSSL_INCLUDE_DIR})
#  target_link_libraries(${TARGET_ORB_SLAM3} ${OPENSSL_CRYPTO_LIBRARY})
#endif()

# OpenCV
find_package(OpenCV 4.0 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3.0 QUIET)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 3.0 not found.")
  endif()
endif()
message(STATUS "OpenCV VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${OpenCV_LIBS})

# Eigen
find_package(Eigen3 3.1.0 REQUIRED)
#target_include_directories(${TARGET_ORB_SLAM3} PRIVATE ${EIGEN3_INCLUDE_DIR})
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3_VERSION: ${Eigen3_VERSION}")
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC Eigen3::Eigen)

# Pangolin
find_package(Pangolin REQUIRED)
#target_include_directories(${TARGET_ORB_SLAM3} PRIVATE ${Pangolin_INCLUDE_DIRS})
message(STATUS "Pangolin_LIBRARIES: ${Pangolin_LIBRARIES}")
message(STATUS "Pangolin_VERSION: ${Pangolin_VERSION}")
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${Pangolin_LIBRARIES})

# realsense2
find_package(realsense2)
if(realsense2_FOUND)
  message(STATUS "realsense2_VERSION: ${realsense2_VERSION}")
endif()

# Boost
find_package(Boost REQUIRED COMPONENTS serialization)
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
target_link_libraries(${TARGET_ORB_SLAM3} PRIVATE ${Boost_LIBRARIES})

# Backward
if(CMAKE_BUILD_TYPE MATCHES Debug)
  find_package(Backward)
  message(STATUS "BACKWARD_LIBRARIES: ${BACKWARD_LIBRARIES}")
  target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC Backward::Backward)
endif()


# Build examples
if(ORB_SLAM3_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif()

#Old examples
if(ORB_SLAM3_BUILD_EXAMPLES_OLD)
  add_subdirectory(Examples_old)
endif()






endif()
endif()
endif()

