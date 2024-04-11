cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM3_EXAMPLES LANGUAGES CXX)

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/RGB-D)

add_executable(rgbd_tum
        RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum PRIVATE ${TARGET_ORB_SLAM3})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i
            RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Stereo)

add_executable(stereo_kitti
        Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti PRIVATE ${TARGET_ORB_SLAM3})

add_executable(stereo_euroc
        Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc PRIVATE ${TARGET_ORB_SLAM3})

add_executable(stereo_tum_vi
        Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi PRIVATE ${TARGET_ORB_SLAM3})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265
            Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265 ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})

    add_executable(stereo_realsense_D435i
            Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Monocular)

add_executable(mono_tum
        Monocular/mono_tum.cc)
target_link_libraries(mono_tum PRIVATE ${TARGET_ORB_SLAM3})

add_executable(mono_kitti
        Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti PRIVATE ${TARGET_ORB_SLAM3})

add_executable(mono_euroc
        Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc PRIVATE ${TARGET_ORB_SLAM3})

add_executable(mono_tum_vi
        Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi PRIVATE ${TARGET_ORB_SLAM3})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265
            Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265 ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})

    add_executable(mono_realsense_D435i
            Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc PRIVATE ${TARGET_ORB_SLAM3})

add_executable(mono_inertial_tum_vi
        Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi PRIVATE ${TARGET_ORB_SLAM3})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265
            Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265 ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})

    add_executable(mono_inertial_realsense_D435i
            Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc PRIVATE ${TARGET_ORB_SLAM3})

add_executable(stereo_inertial_tum_vi
        Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi PRIVATE ${TARGET_ORB_SLAM3})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265
            Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265 ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})

    add_executable(stereo_inertial_realsense_D435i
            Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Calibration)
if(realsense2_FOUND)
    add_executable(recorder_realsense_D435i
            Calibration/recorder_realsense_D435i.cc)
    target_link_libraries(recorder_realsense_D435i ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})

    add_executable(recorder_realsense_T265
            Calibration/recorder_realsense_T265.cc)
    target_link_libraries(recorder_realsense_T265 ${TARGET_ORB_SLAM3} ${realsense2_LIBRARY})
endif()
