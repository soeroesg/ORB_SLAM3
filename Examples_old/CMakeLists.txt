cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM3_EXAMPLES_OLD LANGUAGES CXX)

#Old examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/RGB-D)

add_executable(rgbd_tum_old
        RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum_old ${TARGET_ORB_SLAM3})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i_old
            RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i_old ${TARGET_ORB_SLAM3})
endif()
 
 
# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/RGB-D-Inertial)
 
if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i_old
            RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i_old ${TARGET_ORB_SLAM3})
endif()
 
#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Stereo)

add_executable(stereo_kitti_old
        Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti_old ${TARGET_ORB_SLAM3})

add_executable(stereo_euroc_old
        Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc_old ${TARGET_ORB_SLAM3})

add_executable(stereo_tum_vi_old
        Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi_old ${TARGET_ORB_SLAM3})
 
if(realsense2_FOUND)
    add_executable(stereo_realsense_t265_old
            Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265_old ${TARGET_ORB_SLAM3})
 
    add_executable(stereo_realsense_D435i_old
            Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i_old ${TARGET_ORB_SLAM3})
 endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Monocular)

add_executable(mono_tum_old
        Monocular/mono_tum.cc)
target_link_libraries(mono_tum_old ${TARGET_ORB_SLAM3})

add_executable(mono_kitti_old
        Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti_old ${TARGET_ORB_SLAM3})

add_executable(mono_euroc_old
        Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc_old ${TARGET_ORB_SLAM3})

add_executable(mono_tum_vi_old
        Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi_old ${TARGET_ORB_SLAM3})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265_old
            Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265_old ${TARGET_ORB_SLAM3})

    add_executable(mono_realsense_D435i_old
            Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i_old ${TARGET_ORB_SLAM3})
 endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Monocular-Inertial)

add_executable(mono_inertial_euroc_old
        Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc_old ${TARGET_ORB_SLAM3})

add_executable(mono_inertial_tum_vi_old
        Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi_old ${TARGET_ORB_SLAM3})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265_old
            Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265_old ${TARGET_ORB_SLAM3})

    add_executable(mono_inertial_realsense_D435i_old
            Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i_old ${TARGET_ORB_SLAM3})
endif()
 
#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Stereo-Inertial)
add_executable(stereo_inertial_euroc_old
        Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc_old ${TARGET_ORB_SLAM3})

add_executable(stereo_inertial_tum_vi_old
        Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi_old ${TARGET_ORB_SLAM3})
 
if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265_old
            Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265_old ${TARGET_ORB_SLAM3})

    add_executable(stereo_inertial_realsense_D435i_old
            Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i_old ${TARGET_ORB_SLAM3})
endif()