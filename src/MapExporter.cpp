#include "MapExporter.h"
#include "System.h"
#include "KeyFrame.h"
#include "Converter.h"

#include <opencv2/core/utils/filesystem.hpp>
//#include <filesystem> //only available in C++17

#include <iostream>
#include <vector>
#include <utility>

namespace ORB_SLAM3 {

void MapExporter::SaveKeyFrameTrajectoryColmap(const System& ORBSLAM3System, const string& path, const cv::Size& imgSize) {
    std::cout << std::endl << "Saving keyframe trajectory to " << path << " ..." << std::endl;

    // make sure the path exists and it is a directory
    if (!cv::utils::fs::exists(path)) {
        if (!cv::utils::fs::createDirectory(path)) {
            std::string msg = "Could not create directory " + path;
            std::cout << msg << std::endl;
            return;
        }
    }
    if (!cv::utils::fs::isDirectory(path)) {
        std::string msg = path + " is not a directory! Please provide a directory for exporting keyframe trajectory in colmap format.";
        std::cout << msg << std::endl;
        return;
    }

    // Gather all keyframes from all sub-maps
    std::vector<Map*> vpMaps = ORBSLAM3System.mpAtlas->GetAllMaps();
    Map* pLargestMap = nullptr;
    int maxNumKFs = 0;
    for (Map* pMap : vpMaps) {
        if (pMap->GetAllKeyFrames().size() > maxNumKFs) {
            maxNumKFs = pMap->GetAllKeyFrames().size();
            pLargestMap = pMap;
        }
    }
    if (pLargestMap == nullptr) {
        std::cout << "WARNING: the map is empty" << std::endl;
        return;
    }
    std::vector<KeyFrame*> vpKFs = pLargestMap->GetAllKeyFrames();
    std::sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // We extract the camera information from keyframes while iterating through the keyframes
    std::map<unsigned long, GeometricCamera*> cameraPtrs;
    std::map<unsigned long, cv::Size> cameraSizes; // stores image size for each camera ID

    /*
    Example images.txt
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of images: 2, mean observations per image: 2
    1 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1 P1180141.JPG
    2362.39 248.498 58396 1784.7 268.254 59027 1784.7 268.254 -1
    2 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1 P1180142.JPG
    1190.83 663.957 23056 1258.77 640.354 59070
    */
    std::string images_filename = path + "/images.txt";
    std::ofstream images_file;
    images_file.open(images_filename.c_str());
    images_file << std::fixed;
    if (!images_file.is_open()) {
        std::cout << "ERROR: Cannot open file " << images_filename << std::endl;
        return;
    }

    /*
    Though this is normally done by colmap, we store here the image IDs together with the original file names
    */
    std::string image_ids_filename = path + "/image_ids.txt";
    std::ofstream image_ids_file;
    image_ids_file.open(image_ids_filename.c_str());
    if (!image_ids_file.is_open()) {
        std::cout << "ERROR: Cannot open file " << image_ids_filename << std::endl;
        return;
    }

    size_t num_keyframes = 0;
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];

        if (pKF->isBad())
            continue;

        // check whether we have already stored this camera
        GeometricCamera* cameraPtr = pKF->mpCamera;
        unsigned int cameraId = cameraPtr->GetId();
        std::cout << "KeyFrame " << pKF->mnId << " was captured by camera " << cameraId << std::endl;
        if (cameraPtrs.find(cameraId) == cameraPtrs.end()) {
            // we have not seen this camera before
            cameraPtrs.insert(std::make_pair(cameraId, cameraPtr));
            //cv::Size cameraSize = pKF->imgLeft.size();
            cv::Size cameraSize = imgSize; // TODO: this should be stored in KeyFrame instead of passing from outside.
            cameraSizes.insert(std::make_pair(cameraId, cameraSize));
        }

        image_ids_file << pKF->mnId << " " << pKF->mNameFile << std::endl;

        if (ORBSLAM3System.mSensor == System::eSensor::IMU_MONOCULAR ||
            ORBSLAM3System.mSensor == System::eSensor::IMU_STEREO) {

            cv::Mat R = pKF->GetImuRotation().t();
            std::vector<float> q = Converter::toQuaternion(R);
            cv::Mat twb = pKF->GetImuPosition();
            // warning: q.w comes first!
            images_file << pKF->mnId
                << " " << setprecision(6)
                << " " << q[3] << " " << q[0] << " " << q[1] << " " << q[2]
                << " " << twb.at<float>(0) << " " << twb.at<float>(1) << " " << twb.at<float>(2) << std::endl;
        }
        else {
            cv::Mat R = pKF->GetRotation();
            std::vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            // warning: q.w comes first!
            images_file << pKF->mnId
                << " " << setprecision(6)
                << " " << q[3] << " " << q[0] << " " << q[1] << " " << q[2]
                << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << std::endl;
        }

        // leave one empty line after each image
        images_file << std::endl;
        num_keyframes++;
    }
    images_file.close();
    image_ids_file.close();
    std::cout << "Exported " << num_keyframes << " keyframes" << std::endl;


    /*
    Example cameras.txt
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: 3
    1 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
    2 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
    3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531
    */
    std::string cameras_filename = path + "/cameras.txt";
    std::ofstream cameras_file;
    cameras_file.open(cameras_filename.c_str());
    if (!cameras_file.is_open()) {
        std::cout << "ERROR: Cannot open file " << cameras_filename << std::endl;
        return;
    }
    size_t num_cameras = 0;
    for (auto const& it : cameraPtrs) {
        GeometricCamera* cameraPtr = it.second;
        unsigned int cameraId = cameraPtr->GetId();
        cameras_file << cameraId;
        assert(cameraSizes.find(cameraId) != cameraSizes.end());
        // see https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
        if (cameraPtr->GetType() == cameraPtr->CAM_PINHOLE) {
            // see CameraModels/Pinhole.h
            cameras_file << " " << "PINHOLE";
            cameras_file << " " << cameraSizes[cameraId].width;
            cameras_file << " " << cameraSizes[cameraId].height;
            // this model has fx, fy, cx, cy
            float fx = static_cast<Pinhole*>(cameraPtr)->toK().at<float>(0, 0);
            float fy = static_cast<Pinhole*>(cameraPtr)->toK().at<float>(1, 1);
            float cx = static_cast<Pinhole*>(cameraPtr)->toK().at<float>(0, 2);
            float cy = static_cast<Pinhole*>(cameraPtr)->toK().at<float>(1, 2);
            cameras_file << " " << fx << " " << fy << " " << cx << " " << cy;
        }
        else if (cameraPtr->GetType() == cameraPtr->CAM_FISHEYE) {
            // see CameraModels/KannalaBrandt8.h
            cameras_file << " " << "OPENCV_FISHEYE";
            cameras_file << " " << cameraSizes[cameraId].width;
            cameras_file << " " << cameraSizes[cameraId].height;
            // this model has fx, fy, cx, cy, k1, k1, k2, k4
            float fx = static_cast<KannalaBrandt8*>(cameraPtr)->toK().at<float>(0, 0);
            float fy = static_cast<KannalaBrandt8*>(cameraPtr)->toK().at<float>(1, 1);
            float cx = static_cast<KannalaBrandt8*>(cameraPtr)->toK().at<float>(0, 2);
            float cy = static_cast<KannalaBrandt8*>(cameraPtr)->toK().at<float>(1, 2);
            float k1 = cameraPtr->getParameter(4);
            float k2 = cameraPtr->getParameter(5);
            float k3 = cameraPtr->getParameter(6);
            float k4 = cameraPtr->getParameter(7);
            cameras_file << " " << fx << " " << fy << " " << cx << " " << cy;
            cameras_file << " " << k1 << " " << k2 << " " << k3 << " " << k4;
        }
        else {
            throw std::runtime_error("Unexpected camera type");
        }
        cameras_file << std::endl;
        num_cameras++;
    }
    std::cout << "Exported " << num_cameras << " cameras" << std::endl;

    /*
    Example points3D.txt
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # Number of points: 3, mean track length: 3.3334
    63390 1.67241 0.292931 0.609726 115 121 122 1.33927 16 6542 15 7345 6 6714 14 7227
    63376 2.01848 0.108877 -0.0260841 102 209 250 1.73449 16 6519 15 7322 14 7212 8 3991
    63371 1.71102 0.28566 0.53475 245 251 249 0.612829 118 4140 117 4473
    */
    std::string points3D_filename = path + "/points3D.txt";
    std::ofstream points3D_file;
    points3D_file.open(points3D_filename.c_str());
    if (!points3D_file.is_open()) {
        std::cout << "ERROR: Cannot open file " << points3D_filename << std::endl;
        return;
    }
    std::cout << "Exported " << 0 << " 3D points" << std::endl;


    // TODO: export neighboring keyframes as image pairs that should be matched
    // ...

}

} // namespace ORB_SLAM3
