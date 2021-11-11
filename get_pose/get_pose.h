#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <memory>

class camera
{
public:
    float fix_k[5];
    cv::Mat_<double> camera_matrix;
    cv::Mat_<double> distCoeffes;
    cv::Mat_<double> camera_to_gripper;

    void get_parameters(std::string);
    
};

typedef std::shared_ptr<camera> camera_ptr;

//(-13,0,8),(0,-13,8),(13,0,8),(0,13,8)
class object
{
public:
    std::vector<cv::Point2f> corners2D;
    std::vector<cv::Point3f> corners3D;
    cv::Point2f center2D;
    cv::Point3f center3D;

    object()
    {
        corners2D.clear();
        corners3D.clear();
        corners3D.push_back(cv::Point3f(-12.5, 10.5, 0));
        corners3D.push_back(cv::Point3f(-12.5, -10.5, 0));
        corners3D.push_back(cv::Point3f(12.5, -10.5, 0));
        corners3D.push_back(cv::Point3f(12.5, 10.5, 0));
        // corners3D.push_back(cv::Point3f(-22.5, 17.5, 0));
        // corners3D.push_back(cv::Point3f(-22.5, -17.5, 0));
        // corners3D.push_back(cv::Point3f(22.5, -17.5, 0));
        // corners3D.push_back(cv::Point3f(22.5, 17.5, 0));
    }
    bool objectDetector(cv::Mat &image);
    cv::Mat get_pose(camera_ptr cam, bool flag = false);
    cv::Mat object_in_gripper_coordinate(camera_ptr cam, cv::Point3f pt);
};

typedef std::shared_ptr<object> object_ptr;