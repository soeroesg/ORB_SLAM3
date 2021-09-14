#include "MapExporter.h"
#include "System.h"
#include "KeyFrame.h"
#include "Converter.h"

#include <iostream>
#include <vector>
#include <utility>
#include <set>
#include <map>

#include <opencv2/core/version.hpp>
#if(__cplusplus >= 201703L)
  #define OS3_USE_STD_FILESYSTEM 1
  #include <filesystem> //only available in C++17
#elif(CV_VERSION_MAJOR >= 4)
  #define OS3_USE_CV_FILESYSTEM 1
  #include <opencv2/core/utils/filesystem.hpp> // only available in OpenCV 3.4
#else
  #define OS3_USE_BOOST_FILESYSTEM 1
  #include <boost/filesystem.hpp>
#endif


namespace ORB_SLAM3 {

// check that the parameters stored in the KeyFrame are the same as the ones stored in the camera
bool validateCameraParameters(KeyFrame* pKF, GeometricCamera* cameraPtr) {
    bool isCorrect = true;
    if (cameraPtr->getParameter(0) != pKF->fx) {
        std::cout << "WARNING: camera parameter mismatch: fx" << std::endl; isCorrect = false;
    }
    if (cameraPtr->getParameter(1) != pKF->fy) {
        std::cout << "WARNING: camera parameter mismatch: fy" << std::endl; isCorrect = false;
    }
    if (cameraPtr->getParameter(2) != pKF->cx) {
        std::cout << "WARNING: camera parameter mismatch: cx" << std::endl; isCorrect = false;
    }
    if (cameraPtr->getParameter(3) != pKF->cy) {
        std::cout << "WARNING: camera parameter mismatch: cy" << std::endl; isCorrect = false;
    }
    // NOTE: the Pinhole model does not store the distortion parameters (why??), so we can stop here
    if (cameraPtr->GetType() == cameraPtr->CAM_PINHOLE
        && cameraPtr->size() < 5) {
        return isCorrect;
    }

    for (int i = 0; i < pKF->mDistCoef.total(); i++) {
        if (cameraPtr->size() < 5 + i) {
            std::cout << "WARNING: camera distortion parameter " << i << " is missing" << std::endl;
        }
        float d_kf = pKF->mDistCoef.at<float>(i);
        float d_cam = cameraPtr->getParameter(4 + i);
        if (d_cam != d_kf) {
            std::cout << "WARNING: camera distortion parameter "
                << i << " mismatch: " << d_cam << " vs " << d_kf << std::endl;
            isCorrect = false;
        }
    }
    return isCorrect;
}

void MapExporter::SaveKeyFrameTrajectoryColmap(const System& ORBSLAM3System, const string& path, bool bCopyImages/* = false*/) {
    std::cout << std::endl << "Saving keyframe trajectory to " << path << " ..." << std::endl;

    // make sure the path exists and it is a directory
#if(OS3_USE_STD_FILESYSTEM)
    if (!std::filesystem::exists(path)) {
        if (!std::filesystem::create_directory(path)) {
#elif(OS3_USE_CV_FILESYSTEM)
    if (!cv::utils::fs::exists(path)) {
        if (!cv::utils::fs::createDirectory(path)) {
#elif(OS3_USE_BOOST_FILESYSTEM)
    if (!boost::filesystem::exists(path)) {
        if (!boost::filesystem::create_directory(path)) {
#endif
            std::string msg = "Could not create directory " + path;
            std::cout << msg << std::endl;
            return;
        }
    }
#if(OS3_USE_STD_FILESYSTEM)
    if (!std::filesystem::is_directory(path)) {
#elif(OS3_USE_CV_FILESYSTEM)
    if (!cv::utils::fs::isDirectory(path)) {
#elif(OS3_USE_BOOST_FILESYSTEM)
    if (!boost::filesystem::is_directory(path)) {
#endif
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
    std::map<unsigned long, cv::Size> cameraSizes; // stores image size for each camera ID, because the camera models don't store it
    std::map<unsigned long, cv::Mat> cameraDistortions; // stores distortion parameters for each camera ID, because the camera models don't store them

    // Though image IDs are normally given by colmap, we store here the image IDs given by ORBSLAM together with the original file names
    std::map<unsigned int, std::string> image_ids;

    // We also collect potentially matching images: the neighboring keyframes.
    std::set<std::pair<unsigned int, unsigned int>> matches;

    /*
    Example images.txt
    See https://colmap.github.io/format.html#images-txt
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

    size_t num_keyframes = 0;
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];

        if (pKF->isBad())
            continue;

        unsigned long keyFrameId = pKF->mnId;
        std::string keyFrameFileName = pKF->mNameFile;

        // check whether we have already stored this camera
        GeometricCamera* cameraPtr = pKF->mpCamera;
        unsigned int cameraId = cameraPtr->GetId(); // NOTE we could increase IDs by 1 because colmap does not like ID 0
        std::cout << "KeyFrame " << keyFrameId << " was captured by camera " << cameraId << std::endl;
        validateCameraParameters(pKF, cameraPtr);
        if (cameraPtrs.find(cameraId) == cameraPtrs.end()) {
            // we have not seen this camera before
            cameraPtrs.insert(std::make_pair(cameraId, cameraPtr));

            // NOTE: the original ORB_SLAM3 does not store the input image so this is empty.
            // however, our modified version has a setting for storing the input images within the keyframes.
            cv::Size cameraSize = pKF->imageSize;
            if (cameraSize.empty()) {
                std::cout << "WARNING: camera " + std::to_string(cameraId) + " seems to have 0 image size" << std::endl;
            }
            cameraSizes.insert(std::make_pair(cameraId, cameraSize));
            cv::Mat cameraDisortion = pKF->mDistCoef.clone();
            cameraDistortions.insert(std::make_pair(cameraId, cameraDisortion));
        }

        // NOTE:
        // ORBSLAM coordinate system is X to the right, Y down, Z forward
        // Colmap cooridnate system: X to the right, Y down, Z forward
        // But ORBSLAM stores the world pose in camera frame, but colmap wants the camera in world frame,
        // therefore we need to inverse the poses
        if (ORBSLAM3System.mSensor == System::eSensor::IMU_MONOCULAR ||
            ORBSLAM3System.mSensor == System::eSensor::IMU_STEREO) {

            cv::Mat R_c = pKF->GetImuRotation().t(); // NOTE: the other exporters also take the inverse here when IMU pose is used. why?
            assert(Converter::isRotationMatrix(R_c));
            cv::Mat R_c_inv = R_c.t(); // inverse is just the transpose
            std::vector<float> q = Converter::toQuaternion(R_c_inv);
            cv::Mat t_c = pKF->GetImuPosition();
            cv::Mat t = (-1) * (R_c_inv * t_c);
            // warning: q.w comes first!
            images_file << keyFrameId
                << setprecision(6)
                << " " << q[3] << " " << q[0] << " " << q[1] << " " << q[2]
                << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2);
        }
        else {
            cv::Mat R_c = pKF->GetRotation();
            assert(Converter::isRotationMatrix(R_c));
            cv::Mat R_c_inv = R_c.t(); // inverse is just the transpose
            std::vector<float> q = Converter::toQuaternion(R_c_inv);
            cv::Mat t_c = pKF->GetCameraCenter();
            cv::Mat t = (-1) * (R_c_inv * t_c);
            // warning: q.w comes first!
            images_file << keyFrameId
                << setprecision(6)
                << " " << q[3] << " " << q[0] << " " << q[1] << " " << q[2]
                << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2);
        }

        if (bCopyImages) {
            std::string path_images = path + "/images";
#if(OS3_USE_STD_FILESYSTEM)
            if (!std::filesystem::exists(path_images)) {
                if (!std::filesystem::create_directory(path_images)) {
#elif(OS3_USE_CV_FILESYSTEM)
            if (!cv::utils::fs::exists(path_images)) {
                if (!cv::utils::fs::createDirectory(path_images)) {
#elif(OS3_USE_BOOST_FILESYSTEM)
            if (!boost::filesystem::exists(path_images)) {
                if (!boost::filesystem::create_directory(path_images)) {
#endif
                    std::cout << "Could not create directory " + path_images << std::endl;
                }
            }
            std::string keyFrameFileNameBase = keyFrameFileName.substr(keyFrameFileName.find_last_of("/\\") + 1);
            std::string keyFrameFilenameNew = path_images + "/" + keyFrameFileNameBase;
            std::ifstream source(keyFrameFileName, ios::binary);
            std::ofstream dest(keyFrameFilenameNew, ios::binary);
            dest << source.rdbuf();
            source.close();
            dest.close();
            //keyFrameFileName = keyFrameFilenameNew; // WARNING: we overwrite the file name to point to the new copy
            keyFrameFileName = keyFrameFileNameBase; // WARNING: only store the base filename within the 'images' folder
        }

        images_file << " " << cameraId << " " << keyFrameFileName << std::endl;

        // leave one empty line after each image
        images_file << std::endl;

        // save image ID
        image_ids.insert(std::make_pair(keyFrameId, keyFrameFileName));

        // now check connected keyframes, these will be good candidates for image matching
        std::set<KeyFrame*> connectedKeyFrames = pKF->GetConnectedKeyFrames();
        for (auto connectedKeyFrame : connectedKeyFrames) {
            // Instead of storing filenames directly here (which would lead to storing each match twice, once in each direction),
            // we store the ids but in an ordered way. We will look up the filenames that belong to the IDs later.
            // The set will take care of overwriting duplicate entries about matches.
            unsigned int smallerId = std::min(keyFrameId, connectedKeyFrame->mnId);
            unsigned int largerId = std::max(keyFrameId, connectedKeyFrame->mnId);
            matches.insert(std::make_pair(smallerId, largerId));
        }

        num_keyframes++;
    }
    images_file.close();
    std::cout << "Exported " << num_keyframes << " keyframes" << std::endl;

    /*
    * Export the image IDs (given by ORBSLAM) together with the corresponding file names
    */
    std::string image_ids_filename = path + "/image_ids_orbslam.txt";
    std::ofstream image_ids_file;
    image_ids_file.open(image_ids_filename.c_str());
    if (!image_ids_file.is_open()) {
        std::cout << "ERROR: Cannot open file " << image_ids_filename << std::endl;
        return;
    }
    for (auto image_id : image_ids) {
        image_ids_file << image_id.first << " " << image_id.second << std::endl;
    }
    image_ids_file.close();

    /*
    Example cameras.txt
    See https://colmap.github.io/format.html#cameras-txt
    # Camera list with one line of data per camera:
    #   (prior_focal_length is a boolean as integer, and this entry was missing from earlier colmap versions)
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[], PRIOR_FOCAL_LENGTH
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
        unsigned int cameraId = cameraPtr->GetId(); // NOTE we could increase IDs by 1 because colmap does not like ID 0
        cameras_file << cameraId;
        assert(cameraSizes.find(cameraId) != cameraSizes.end());
        // see https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
        if (cameraPtr->GetType() == cameraPtr->CAM_PINHOLE) {
            // see CameraModels/Pinhole.h
            float fx = static_cast<Pinhole*>(cameraPtr)->toK().at<float>(0, 0);
            float fy = static_cast<Pinhole*>(cameraPtr)->toK().at<float>(1, 1);
            float cx = static_cast<Pinhole*>(cameraPtr)->toK().at<float>(0, 2);
            float cy = static_cast<Pinhole*>(cameraPtr)->toK().at<float>(1, 2);

            // WARNING: this model does not store the distortion parameters! why?
            // however, the keyframe stores k1, k1, p1, p2, k3
            cv::Mat cameraDistortion = cameraDistortions[cameraId]; // so we store them separately
            if (cv::norm(cameraDistortion) < 0.0001) { // all entries very close to zero
                cameras_file << " " << "PINHOLE";
                // this Colmap camera model has fx, fy, cx, cy
                cameras_file << " " << cameraSizes[cameraId].width;
                cameras_file << " " << cameraSizes[cameraId].height;
                cameras_file << setprecision(6);
                cameras_file << " " << fx << " " << fy << " " << cx << " " << cy;
            } else if (cameraDistortion.total() == 4 ||
                (cameraDistortion.total() == 5 && cameraDistortion.at<float>(4) == 0.0) ) {
                cameras_file << " " << "OPENCV";
                // this Colmap camera model has fx, fy, cx, cy, k1, k2, p1, p2, (and ignores k3)
                cameras_file << " " << cameraSizes[cameraId].width;
                cameras_file << " " << cameraSizes[cameraId].height;
                cameras_file << " " << fx << " " << fy << " " << cx << " " << cy;
                float k1 = cameraDistortion.at<float>(0);
                float k2 = cameraDistortion.at<float>(1);
                float p1 = cameraDistortion.at<float>(2);
                float p2 = cameraDistortion.at<float>(3);
                cameras_file << " " << k1 << " " << k2 << " " << p1 << " " << p2;
            } else {
                cameras_file << " " << "FULL_OPENCV";
                // this Colmap camera model has fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
                cameras_file << " " << cameraSizes[cameraId].width;
                cameras_file << " " << cameraSizes[cameraId].height;
                cameras_file << " " << fx << " " << fy << " " << cx << " " << cy;
                for (int d_idx = 0; d_idx < 8; d_idx++) {
                    cameras_file << " " << (d_idx < cameraDistortion.total()) ? cameraDistortion.at<float>(d_idx) : 0.0;
                }
            }
        }
        else if (cameraPtr->GetType() == cameraPtr->CAM_FISHEYE) {
            // see CameraModels/KannalaBrandt8.h
            cameras_file << " " << "OPENCV_FISHEYE";
            // this Colmap camera model has fx, fy, cx, cy, k1, k1, k2, k4
            cameras_file << " " << cameraSizes[cameraId].width;
            cameras_file << " " << cameraSizes[cameraId].height;
            float fx = static_cast<KannalaBrandt8*>(cameraPtr)->toK().at<float>(0, 0);
            float fy = static_cast<KannalaBrandt8*>(cameraPtr)->toK().at<float>(1, 1);
            float cx = static_cast<KannalaBrandt8*>(cameraPtr)->toK().at<float>(0, 2);
            float cy = static_cast<KannalaBrandt8*>(cameraPtr)->toK().at<float>(1, 2);
            cameras_file << " " << fx << " " << fy << " " << cx << " " << cy;
            float k1 = cameraPtr->getParameter(4);
            float k2 = cameraPtr->getParameter(5);
            float k3 = cameraPtr->getParameter(6);
            float k4 = cameraPtr->getParameter(7);
            cameras_file << " " << k1 << " " << k2 << " " << k3 << " " << k4;
        }
        else {
            throw std::runtime_error("Unexpected camera type");
        }

        // prior_focal_length (1/0 for true/false) - this tells colmap that the camera parameters are already quite good
        cameras_file << " " << int(1);

        cameras_file << std::endl;
        num_cameras++;
    }
    std::cout << "Exported " << num_cameras << " cameras" << std::endl;

    /*
    Example points3D.txt
    See https://colmap.github.io/format.html#points3d-txt
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

    /*
    * Export matches.txt, neighboring keyframes as image pairs that should be matched
    */
    std::string matches_filename = path + "/matches.txt";
    std::ofstream matches_file;
    matches_file.open(matches_filename.c_str());
    if (!matches_file.is_open()) {
        std::cout << "ERROR: Cannot open file " << matches_filename << std::endl;
        return;
    }
    size_t num_matches = 0;
    for (auto match : matches) {
        std::string first_filename = image_ids[match.first];
        std::string second_filename = image_ids[match.second];
        matches_file << first_filename << " " << second_filename << std::endl;
        num_matches++;
    }
    std::cout << "Exported " << num_matches << " matches (pairs of images)" << std::endl;
}

} // namespace ORB_SLAM3
