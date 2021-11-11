#include "get_pose.h"
using namespace std;
using namespace cv;

#define max_num 255
#define threshold_low 3000
#define threshold_high 35000

void camera::get_parameters(std::string fileName)
{
    cout << fileName << endl;
    cv::FileStorage fs(fileName, cv::FileStorage::READ);

    std::string date;
    fs["calibration_Time"] >> date;

    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> distCoeffes;

    cout << date << camera_matrix << distCoeffes << endl;
}

void trackbar1(int, void *)
{
}
void trackbar2(int, void *)
{
}
void trackbar3(int, void *)
{
}
void trackbar4(int, void *)
{
}
void trackbar5(int, void *)
{
}
void trackbar6(int, void *)
{
}

int h1 = 0, s1 = 0, v1 = 0, h2 = 0, s2 = 0, v2 = 0;

/**
 * @brief get the four 2D corners and one center of the EXTERNAL RotatedRect of ellipse
 * 
 * @param image 
 */
bool object::objectDetector(cv::Mat &image)
{
    Mat src, dst1, dst2, dst;
    bool flag = false;

    cvtColor(image, src, COLOR_BGR2HSV);
    // namedWindow("win");

    // createTrackbar("h1", "win", &h1, max_num, trackbar1);
    // createTrackbar("s1", "win", &s1, max_num, trackbar2);
    // createTrackbar("v1", "win", &v1, max_num, trackbar3);

    // createTrackbar("h2", "win", &h2, max_num, trackbar4);
    // createTrackbar("s2", "win", &s2, max_num, trackbar5);
    // createTrackbar("v2", "win", &v2, max_num, trackbar6);

    // inRange(src, Scalar(0, 40, 57), Scalar(214, 255, 255), dst1); //red
    // inRange(src, Scalar(0, 23, 76), Scalar(151, 197, 255), dst2); //red
    // dst = dst1 - dst2;

    inRange(src, Scalar(81, 34, 13), Scalar(151, 197, 109), dst); //black
    // inRange(src, Scalar(h1, s1, v1), Scalar(h2, s2, v2), dst);
    // dst = dst1 - dst2;
    medianBlur(dst, dst, 3);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dst, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<RotatedRect> minEllipse;

    vector<Point2f> center;
    vector<cv::Point2f> pts;
    cv::Point2f centerPoint;
    minEllipse.clear();
    pts.clear();

    int INDEX = 0;
    bool pattern = 0;
    for (size_t i = 0; i < contours.size(); i++)
    {
        RotatedRect minRect = minAreaRect(contours[i]);

        if (minRect.size.area() > threshold_low && minRect.size.area() < threshold_high)
        //if (contourArea(contours[i]) > threshold_low && contourArea(contours[i]) < threshold_high)
        {
            Point2f ttt[4];
            minRect.points(ttt);
            for (int a = 0; a < 4; a++)
            {
                line(image, ttt[a % 4], ttt[(a + 1) % 4], Scalar(150, 20, 95), 1, 8);
            }

            cv::Point2f left_up, left_down, right_up, right_down;

            int left_up_min = 10000, right_up_max = -10000, left_down_min = 10000, right_down_max = -10000;
            for (auto pt : contours[i])
            {
                if (left_up_min > pt.x + pt.y)
                {
                    left_up_min = pt.x + pt.y;
                    left_up = pt;
                }
                if (right_up_max < pt.y - pt.x)
                {
                    right_up_max = pt.y - pt.x;
                    right_up = pt;
                }
                if (left_down_min > pt.y - pt.x)
                {
                    left_down_min = pt.y - pt.x;
                    left_down = pt;
                }
                if (right_down_max < pt.x + pt.y)
                {
                    right_down_max = pt.x + pt.y;
                    right_down = pt;
                }
            }

            pts.push_back(left_up);
            pts.push_back(left_down);
            pts.push_back(right_down);
            pts.push_back(right_up);

            float L1 = (sqrt((left_up.x - right_up.x) * (left_up.x - right_up.x) + (left_up.y - right_up.y) * (left_up.y - right_up.y)) +
                        sqrt((left_down.x - right_down.x) * (left_down.x - right_down.x) + (left_down.y - right_down.y) * (left_down.y - right_down.y))) /
                       2;
            float L2 = (sqrt((left_down.x - left_up.x) * (left_down.x - left_up.x) + (left_down.y - left_up.y) * (left_down.y - left_up.y)) +
                        sqrt((right_down.x - right_up.x) * (right_down.x - right_up.x) + (right_down.y - right_up.y) * (right_down.y - right_up.y))) /
                       2;
            if (max(L1 / L2, L2 / L1) < 2)
            {
                centerPoint = (left_up + left_down + right_up + right_down) / 4;
                if (center.size() != 0)
                {
                    bool judge_continue = false;
                    for (Point2f c : center)
                    {
                        float distance = (c.x - centerPoint.x) * (c.x - centerPoint.x) +
                                         (c.y - centerPoint.y) * (c.y - centerPoint.y);
                        if (distance < 300)
                        {
                            judge_continue = true;
                            break;
                        }
                    }
                }

                center.push_back(centerPoint);

                // line(image, left_up, left_down, Scalar(150, 20, 95), 1, 8); //画矩形四条边
                // line(image, left_down, right_down, Scalar(150, 20, 95), 1, 8);
                // line(image, right_down, right_up, Scalar(150, 20, 95), 1, 8);
                // line(image, right_up, left_up, Scalar(150, 20, 95), 1, 8);
            }
        }
    }
    // cout << center.size() << endl;

    if (pts.size() == 4)
    {
        vector<Point2f> point;

        //左上 左下 右下 右上
        corners2D.push_back(pts[0]);
        corners2D.push_back(pts[1]);
        corners2D.push_back(pts[2]);
        corners2D.push_back(pts[3]);

        circle(image, corners2D[0], 3, Scalar(150, 120, 125), 3);
        circle(image, corners2D[1], 3, Scalar(150, 120, 125), 3);
        circle(image, corners2D[2], 3, Scalar(150, 120, 125), 3);
        circle(image, corners2D[3], 3, Scalar(150, 120, 125), 3);

        //cout << endl;
        center2D = centerPoint;
        flag = true;
    }

    imshow("hsv", dst);

    imshow("dealing", image);
    return flag;
}

/**
 * @brief get the transform matrix from the world coordinate to the camera coordinate
 * 
 * @param cam 
 * @return Mat 
 */
Mat object::get_pose(camera_ptr cam, bool flag)
{
    Mat rvec;
    Mat tvec;
    Mat transform_matrix;

    if (flag)
    {
        solvePnP(corners3D, corners2D, cam->camera_matrix, cam->distCoeffes, rvec, tvec);

        Rodrigues(rvec, rvec); //旋转矩阵

        // cout << tvec << endl;
        // cout << sqrt(tvec.at<double>(0, 0) * tvec.at<double>(0, 0) +
        //              tvec.at<double>(0, 1) * tvec.at<double>(0, 1) +
        //              tvec.at<double>(0, 2) * tvec.at<double>(0, 2))
        //      << endl;
        transform_matrix = (Mat_<double>(4, 4) << rvec.at<double>(0, 0), rvec.at<double>(0, 1), rvec.at<double>(0, 2), tvec.at<double>(0, 0),
                            rvec.at<double>(1, 0), rvec.at<double>(1, 1), rvec.at<double>(1, 2), tvec.at<double>(0, 1),
                            rvec.at<double>(2, 0), rvec.at<double>(2, 1), rvec.at<double>(2, 2), tvec.at<double>(0, 2),
                            0.0, 0.0, 0.0, 1.0);

        //print(transform_matrix);
    }
    else
    {
        transform_matrix = Mat::eye(4, 4, CV_8UC1);
    }

    return transform_matrix;
}

cv::Mat object::object_in_gripper_coordinate(camera_ptr cam, cv::Point3f pt)
{
    //    cout<<pt<<endl;
    //    cout<<cam->camera_to_gripper<<endl;
    Mat pts = (Mat_<double>(4, 1) << pt.x / 1000, pt.y / 1000, pt.z / 1000, 1);
    cv::Mat temp = cam->camera_to_gripper * pts; //
    //cout<<"!!!"<<temp<<endl;
    return temp;
}
