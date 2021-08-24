// This helper class exports the ORBSLAM3 map in various formats
// KITTI, EuRoC, TUM exporters have been moved here from ORBSLAM3 System class
// Added Colmap exporter
//
// Created by https://github.com/soeroesg
// 08.08.2021
//
// For further info about colmap, see https://colmap.github.io/faq.html
// Code pieces also taken from https://github.com/tsattler/understanding_apr

#ifndef MAP_EXPORTER_H
#define MAP_EXPORTER_H

#include "System.h"
#include <string>

namespace ORB_SLAM3 {

class MapExporter {
public:
    // Save the keyframes in Colmap-readable format
    // See https://colmap.github.io/format.html
    static void SaveKeyFrameTrajectoryColmap(const System& ORBSLAM3, const std::string& dir_path, bool bCopyImages = false);

};

} // namespace ORB_SLAM3

# endif // MAP_EXPORTER_H
