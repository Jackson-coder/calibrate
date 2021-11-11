#include <iostream>
#include <opencv2/opencv.hpp>
#include "get_pose/get_pose.h"
using namespace std;
using namespace cv;

int main(int, char **)
{
    cv::Mat image;
    cv::VideoCapture capture(0);

    capture.set(CAP_PROP_EXPOSURE, -12); //自动曝光

    auto i = capture.get(CAP_PROP_EXPOSURE);
    cout << "当前曝光为：" << i << endl;
    camera_ptr cam{new camera};
    cam->camera_to_gripper = (Mat_<double>(4, 4) << 0, 0, 1, 0.04,
                              0, -1, 0, 0.05,
                              -1, 0, 0, 0,
                              0, 0, 0, 1);

    cam->get_parameters("/home/lin/robot_camera/camera_pose_estimation/get_pose/out_camera_data.xml");
    cv::Point3f pt;
    while (1)
    {
        capture >> image;
        //cout << cam->camera_matrix << ' ' << cam->distCoeffes << endl;
        if (!image.empty())
        {
            vector<Mat> imageRGB;

            //RGB三通道分离
            split(image, imageRGB);

            //求原始图像的RGB分量的均值
            double R, G, B;
            B = mean(imageRGB[0])[0];
            G = mean(imageRGB[1])[0];
            R = mean(imageRGB[2])[0];

            //需要调整的RGB分量的增益
            double KR, KG, KB;
            KB = (R + G + B) / (3 * B);
            KG = (R + G + B) / (3 * G);
            KR = (R + G + B) / (3 * R);

            //调整RGB三个通道各自的值
            imageRGB[0] = imageRGB[0] * KB;
            imageRGB[1] = imageRGB[1] * KG;
            imageRGB[2] = imageRGB[2] * KR;

            //RGB三通道图像合并
            merge(imageRGB, image);
            imshow("白平衡调整后", image);

            bool flag;
            //cv::imshow("win", image);
            object_ptr obj{new object};
            flag = obj->objectDetector(image);
            Mat transform_matrix = obj->get_pose(cam, flag);
            if (flag)
            {
                pt = Point3f(transform_matrix.at<double>(0, 3), transform_matrix.at<double>(1, 3), transform_matrix.at<double>(2, 3));
                Mat matrix = obj->object_in_gripper_coordinate(cam, pt);
                cout << pt << endl;
                cout << matrix << endl;
            }
            
            cv::waitKey(1);
        }
    }
}

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <math.h>
// using namespace std;
// using namespace cv;

// int main()
// {
//     Mat camera_matrix, distCoeffes;
//     cv::FileStorage fs("/home/lin/robot_camera/camera_pose_estimation/out_camera_data.xml", cv::FileStorage::READ);
//     fs["camera_matrix"] >> camera_matrix;
//     fs["distortion_coefficients"] >> distCoeffes;

//     cv::Mat image;
//     cv::VideoCapture capture(2);

//     int cols = 9;

//     int rows = 6;

//     float distance = 19; //间距30mm

//     cv::Size patternSize(cols, rows);

//     while (1)
//     {
//         capture >> image;
//         if (!image.empty())
//         {
//             std::vector<cv::Point2f> corners;

//             std::vector<std::vector<cv::Point2f>> cornersVect;

//             std::vector<cv::Point3f> worldPoints;

//             std::vector<std::vector<cv::Point3f>> worldPointsVect;

//             vector<cv::Point2f> corners2D;

//             vector<cv::Point3f> corners3D;

//             corners.clear();
//             cornersVect.clear();
//             worldPoints.clear();
//             worldPointsVect.clear();

//             corners3D.push_back(cv::Point3f(-96.5, 67, 0));//21 25
//             corners3D.push_back(cv::Point3f(-96.5, -67, 0));
//             corners3D.push_back(cv::Point3f(96.5, 67, 0));
//             corners3D.push_back(cv::Point3f(96.5, -67, 0));

//             bool find = cv::findChessboardCorners(image, patternSize, corners);

//             if (find)
//             {
//                 int left_up_min = 10000, right_up_max = -10000, left_down_min = 10000, right_down_max = -10000;
//                 cv::Point2f left_up, left_down, right_up, right_down;

//                 for (auto pt : corners)
//                 {
//                     if (left_up_min > pt.x + pt.y)
//                     {
//                         left_up_min = pt.x + pt.y;
//                         left_up = pt;
//                     }
//                     if (right_up_max < pt.y - pt.x)
//                     {
//                         right_up_max = pt.y - pt.x;
//                         right_up = pt;
//                     }
//                     if (left_down_min > pt.y - pt.x)
//                     {
//                         left_down_min = pt.y - pt.x;
//                         left_down = pt;
//                     }
//                     if (right_down_max < pt.x + pt.y)
//                     {
//                         right_down_max = pt.x + pt.y;
//                         right_down = pt;
//                     }
//                 }

//                 corners2D.push_back(left_up);
//                 corners2D.push_back(left_down);
//                 corners2D.push_back(right_up);
//                 corners2D.push_back(right_down);

//                 for (auto center : corners2D)
//                 {
//                     circle(image, center, 10, Scalar(130, 46, 99), 2);
//                 }

//                 cv::Mat rvec, tvec;

//                 solvePnP(corners3D, corners2D, camera_matrix, distCoeffes, rvec, tvec);

//                 // cout << tvec << endl;
//                 cout << sqrt(tvec.at<double>(0, 0) * tvec.at<double>(0, 0) +
//                              tvec.at<double>(1, 0) * tvec.at<double>(1, 0) +
//                              tvec.at<double>(2, 0) + tvec.at<double>(2, 0))
//                      << endl;

//                 cv::drawChessboardCorners(image, patternSize, corners, find);
//             }

//             cv::imshow("picture", image);

//             cv::waitKey(1);
//         }
//     }
// }