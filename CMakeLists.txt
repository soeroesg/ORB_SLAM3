cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM3)

# Build type (default to Release)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Debug postfix
set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")

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
  src/OptimizableTypes.cpp
  src/MLPnPsolver.cpp
  src/GeometricTools.cc
  src/TwoViewReconstruction.cc
  src/Config.cc
  src/Settings.cc
  src/CameraModels/Pinhole.cpp
  src/CameraModels/KannalaBrandt8.cpp
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
  include/OptimizableTypes.h
  include/MLPnPsolver.h
  include/GeometricTools.h
  include/TwoViewReconstruction.h
  include/SerializationUtils.h
  include/Config.h
  include/Settings.h
  include/CameraModels/GeometricCamera.h
  include/CameraModels/Pinhole.h
  include/CameraModels/KannalaBrandt8.h
)

add_library(${TARGET_ORB_SLAM3} ${ORB_SLAM3_LIB_TYPE}
  ${ORB_SLAM3_HEADERS} ${ORB_SLAM3_SOURCES}
)

target_include_directories(${TARGET_ORB_SLAM3} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/CameraModels>
  $<INSTALL_INTERFACE:include/${TARGET_ORB_SLAM3}>
  $<INSTALL_INTERFACE:include/${TARGET_ORB_SLAM3}/CameraModels>
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
  target_include_directories(${TARGET_ORB_SLAM3} PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2>
      $<INSTALL_INTERFACE:include/DBoW2>
  )
else()
  find_package(DBoW2 REQUIRED)
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC DBoW2)

# G2O
if(ORB_SLAM3_USE_OWN_g2o)
  add_subdirectory(Thirdparty/g2o)
  target_include_directories(${TARGET_ORB_SLAM3} PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/g2o>
      $<INSTALL_INTERFACE:include/g2o>
  )
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
  target_include_directories(${TARGET_ORB_SLAM3} PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}Thirdparty/Sophus>
      $<INSTALL_INTERFACE:>
  )
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
  target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${realsense2_LIBRARY}) # TODO: move these to the examples only
  set(ORB_SLAM3_DEPENDENCY_realsense2 "find_dependency(realsense2 REQUIRED)")
endif()

# Boost
find_package(Boost REQUIRED COMPONENTS serialization)
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
target_link_libraries(${TARGET_ORB_SLAM3} PRIVATE ${Boost_LIBRARIES})

# Backward-cpp
if(CMAKE_BUILD_TYPE MATCHES Debug)
  find_package(Backward REQUIRED)
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


# Install

# Set standard installation directories
set(RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
set(LIBRARY_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(ARCHIVE_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDES_DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
set(INCLUDES_INSTALL_DIR ${INCLUDES_DESTINATION}/${TARGET_ORB_SLAM3})
set(CMAKECONFIG_INSTALL_DIR ${LIBRARY_DESTINATION}/cmake/${TARGET_ORB_SLAM3})

# this relative path allows installed files to be relocatable.
file(RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKECONFIG_INSTALL_DIR}"
    "${INCLUDES_INSTALL_DIR}"
)
set(EXPORT_LIB_INC_DIRS "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}")
set(EXPORT_LIB_INC_DIRS ${EXPORT_LIB_INC_DIRS} "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}/CameraModels")

# install subprojects
if(ORB_SLAM3_USE_OWN_DBoW2)
    #install(FILES Thirdparty/DBoW2/lib/libDBoW2.so DESTINATION ${LIBRARY_DESTINATION})
    install(TARGETS DBoW2 DESTINATION ${LIBRARY_DESTINATION})
    # HACK: copy the DBoW2 headers because the own version has no package install
    install(FILES
        Thirdparty/DBoW2/DBoW2/BowVector.h
        Thirdparty/DBoW2/DBoW2/FClass.h
        Thirdparty/DBoW2/DBoW2/FeatureVector.h
        Thirdparty/DBoW2/DBoW2/FORB.h
        Thirdparty/DBoW2/DBoW2/ScoringObject.h
        Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h
        DESTINATION ${INCLUDES_DESTINATION}/DBoW2/DBoW2)
    install(FILES
        FILES Thirdparty/DBoW2/DUtils/Random.h
        FILES Thirdparty/DBoW2/DUtils/Timestamp.h
        DESTINATION ${INCLUDES_DESTINATION}/DBoW2/DUtils
    )
    install(TARGETS DBoW2 EXPORT ${TARGET_ORB_SLAM3}Targets)
    set(EXPORT_LIB_INC_DIRS ${EXPORT_LIB_INC_DIRS} "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}/../DBoW2")
endif()
if(ORB_SLAM3_USE_OWN_g2o)
    #install(FILES Thirdparty/gREQUIRED2o/lib/libg2o.so DESTINATION ${LIBRARY_DESTINATION})
    install(TARGETS g2o DESTINATION ${LIBRARY_DESTINATION})
    # HACK: copy the DBoW2 headers because the own version has no package install
    install(FILES "Thirdparty/g2o/config.h" DESTINATION ${INCLUDES_DESTINATION}/g2o)
    file(GLOB g2o_HEADERS_H "Thirdparty/g2o/g2o/core/*.h")
    file(GLOB g2o_HEADERS_HPP "Thirdparty/g2o/g2o/core/*.hpp")
    install(FILES ${g2o_HEADERS_H} ${g2o_HEADERS_HPP} DESTINATION ${INCLUDES_DESTINATION}/g2o/g2o/core)
    file(GLOB g2o_HEADERS_H "Thirdparty/g2o/g2o/solvers/*.h")
    file(GLOB g2o_HEADERS_HPP "Thirdparty/g2o/g2o/solvers/*.hpp")
    install(FILES ${g2o_HEADERS_H} ${g2o_HEADERS_HPP} DESTINATION ${INCLUDES_DESTINATION}/g2o/g2o/solvers)
    file(GLOB g2o_HEADERS_H "Thirdparty/g2o/g2o/stuff/*.h")
    file(GLOB g2o_HEADERS_HPP "Thirdparty/g2o/g2o/stuff/*.hpp")
    install(FILES ${g2o_HEADERS_H} ${g2o_HEADERS_HPP} DESTINATION ${INCLUDES_DESTINATION}/g2o/g2o/stuff)
    file(GLOB g2o_HEADERS_H "Thirdparty/g2o/g2o/types/*.h")
    file(GLOB g2o_HEADERS_HPP "Thirdparty/g2o/g2o/types/*.hpp")
    install(FILES ${g2o_HEADERS_H} ${g2o_HEADERS_HPP} DESTINATION ${INCLUDES_DESTINATION}/g2o/g2o/types)
    install(TARGETS g2o EXPORT ${TARGET_ORB_SLAM3}Targets)
    set(EXPORT_LIB_INC_DIRS ${EXPORT_LIB_INC_DIRS} "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}/../g2o")
endif()
if(ORB_SLAM3_USE_OWN_Sophus)
    install(TARGETS sophus DESTINATION ${LIBRARY_DESTINATION})
    set(EXPORT_LIB_INC_DIRS ${EXPORT_LIB_INC_DIRS} "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}/..")
endif()

# install headers
file(GLOB ORB_SLAM3_HEADERS "include/*.h")
install(FILES ${ORB_SLAM3_HEADERS} DESTINATION ${INCLUDES_INSTALL_DIR})
file(GLOB ORB_SLAM3_HEADERS "include/CameraModels/*.h")
install(FILES ${ORB_SLAM3_HEADERS} DESTINATION ${INCLUDES_INSTALL_DIR}/CameraModels)

# install libraries
install(TARGETS ${TARGET_ORB_SLAM3}
    EXPORT ${TARGET_ORB_SLAM3}Targets
    RUNTIME DESTINATION ${RUNTIME_DESTINATION}
    LIBRARY DESTINATION ${LIBRARY_DESTINATION}
    ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
    INCLUDES DESTINATION ${INCLUDES_DESTINATION}
)

# write package config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${TARGET_ORB_SLAM3}ConfigVersion.cmake"
    VERSION 1.0
    COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/ORB_SLAM3Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${TARGET_ORB_SLAM3}Config.cmake"
    INSTALL_DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)
install(EXPORT ${TARGET_ORB_SLAM3}Targets
    DESTINATION ${CMAKECONFIG_INSTALL_DIR})
install(FILES "${PROJECT_BINARY_DIR}/${TARGET_ORB_SLAM3}ConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/${TARGET_ORB_SLAM3}Config.cmake"
    DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)


# TODO: extract and copy the vocabulary into the bin folder
