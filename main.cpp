#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include <unordered_map>
#include <random>
using namespace std;
using namespace cv;

/**
 * @brief 模式功能选项
 * 模式0：双目标定
 * 模式1：单目鱼眼标定
 * 模式2：双目测距
 * 模式3：单目鱼眼转单目rgb
 */
#define pFLAG 0

void help(char **argv)
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: %s <left_image> <right_image> [--algorithm=bm|sgbm|hh|hh4|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
           "[--no-display] [--color] [-o=<disparity_image>] [-p=<point_cloud_file>]\n",
           argv[0]);
}


// unordered_map<cv::Point2f,int,createhash,isEqual> a;

int main(int argc, char **argv)
{
#ifndef pFLAG
    cout << "NO DEFINITION OF PATTERN!!!" << endl;

#elif pFLAG == 0

    std::string img1_filename = "";
    std::string img2_filename = "";
    std::string intrinsic_filename = "";
    std::string extrinsic_filename = "";
    std::string disparity_filename = "";
    std::string point_cloud_filename = "";

    enum
    {
        STEREO_BM = 0,
        STEREO_SGBM = 1,
        STEREO_HH = 2,
        STEREO_VAR = 3,
        STEREO_3WAY = 4,
        STEREO_HH4 = 5
    };
    int alg = STEREO_SGBM;
    int SADWindowSize, numberOfDisparities;
    bool no_display;
    bool color_display;
    float scale;

    Ptr<StereoBM> bm = StereoBM::create(16, 9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
    cv::CommandLineParser parser(argc, argv,
                                 "{@arg1|/home/camera_calibrate2/left13.png|}{@arg2|/home/camera_calibrate2/right13.jpg|}{w|11|}{h|8|}{ws|11|}{help h||}{algorithm|bm|}{max-disparity|16|}{blocksize|5|}{no-display||}{color||}{scale|1|}{i|/home/camera_calibrate2/intrinsics.yml|}{e|/home/camera_calibrate2/extrinsics.yml|}{o|record.jpg|}{p|record|}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    img1_filename = samples::findFile(parser.get<std::string>(0));
    img2_filename = samples::findFile(parser.get<std::string>(1));
    
    if (parser.has("algorithm"))
    {
        std::string _alg = parser.get<std::string>("algorithm");
        alg = _alg == "bm" ? STEREO_BM : _alg == "sgbm" ? STEREO_SGBM : _alg == "hh" ? STEREO_HH : _alg == "var" ? STEREO_VAR : _alg == "hh4" ? STEREO_HH4 : _alg == "sgbm3way" ? STEREO_3WAY : -1;
    }
    numberOfDisparities = parser.get<int>("max-disparity");
    SADWindowSize = parser.get<int>("blocksize");
    scale = parser.get<float>("scale");
    no_display = parser.has("no-display");
    color_display = parser.has("color");
    Size boardSize;
    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");
    int winSize;
    winSize = parser.get<int>("ws");

    if (parser.has("i"))
        intrinsic_filename = parser.get<std::string>("i");
    if (parser.has("e"))
        extrinsic_filename = parser.get<std::string>("e");
    if (parser.has("o"))
        disparity_filename = parser.get<std::string>("o");
    if (parser.has("p"))
        point_cloud_filename = parser.get<std::string>("p");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    if (alg < 0)
    {
        printf("Command-line parameter error: Unknown stereo algorithm\n\n");
        help(argv);
        return -1;
    }
    if (numberOfDisparities < 1 || numberOfDisparities % 16 != 0)
    {
        printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
        help(argv);
        return -1;
    }
    if (scale < 0)
    {
        printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
        return -1;
    }
    if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
    {
        printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
        return -1;
    }
    if (img1_filename.empty() || img2_filename.empty())
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }
    if ((!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()))
    {
        printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }

    if (extrinsic_filename.empty() && !point_cloud_filename.empty())
    {
        printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
        return -1;
    }

    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    cout << img1_filename << ' ' << img2_filename << endl;
    
    if (img1.empty())
    {
        printf("Command-line parameter error: could not load the first input image file\n");
        return -1;
    }
    if (img2.empty())
    {
        printf("Command-line parameter error: could not load the second input image file\n");
        return -1;
    }


    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    

    Size img_size = img2.size();

    Mat M1, D1, M2, D2;

    Mat R, T, R1, P1, R2, P2;

    vector<Point2f> pointbuf, _pointbuf, pointbuf1, pointbuf2;
    typedef Vec<int16_t, 2> Vec2C;
    Mat image1, image2;

    if (!intrinsic_filename.empty())
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        
        if (!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            return -1;
        }

        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, FileStorage::READ);
        if (!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename.c_str());
            return -1;
        }

        fs["R"] >> R;
        fs["T"] >> T;
   
        Mat img11, img22;
        undistort(img1, img11, M1, D1);
        undistort(img2, img22, M2, D2);

        R1 = Mat::eye(Size(3, 3), CV_32FC1);
        R2 = Mat::eye(Size(3, 3), CV_32FC1);

        Mat RT1, RT2;
        Mat a = Mat::eye(3, 3, CV_64F);
        Mat b = (Mat_<double>(3, 1) << 0, 0, 0);
        hconcat(a, b, RT1);
        hconcat(R, T, RT2);
        P1 = M1 * RT1;
        P2 = M2 * RT2;

        bool found, _found, found1, found2;

        found = findChessboardCorners(img1, boardSize, pointbuf,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        _found = findChessboardCorners(img2, boardSize, _pointbuf,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        found1 = findChessboardCorners(img11, boardSize, pointbuf1,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        found2 = findChessboardCorners(img22, boardSize, pointbuf2,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        // improve the found corners' coordinate accuracy
        if (found)
        {
            cornerSubPix(img1, pointbuf, Size(winSize, winSize),
                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
            drawChessboardCorners(img1, boardSize, Mat(pointbuf), found);
        }
        if (_found)
        {
            cornerSubPix(img2, _pointbuf, Size(winSize, winSize),
                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
            drawChessboardCorners(img2, boardSize, Mat(_pointbuf), found);
        }
        if (found1)
        {
            cornerSubPix(img11, pointbuf1, Size(winSize, winSize),
                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
            drawChessboardCorners(img11, boardSize, Mat(pointbuf1), found1);
        }
        if (found2)
        {
            cornerSubPix(img22, pointbuf2, Size(winSize, winSize),
                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
            drawChessboardCorners(img22, boardSize, Mat(pointbuf2), found2);
        }

        Mat points4D;
        vector<Point3f> points3D;
        triangulatePoints(P1, P2, pointbuf1, pointbuf2, points4D);
        cout<<points4D.size()<<endl;
        for(int d=0; d<pointbuf1.size(); d++)
        {
            Point3f point3D;
            point3D.x = points4D.at<float>(0,d)/points4D.at<float>(3,d);
            point3D.y = points4D.at<float>(1,d)/points4D.at<float>(3,d);
            point3D.z = points4D.at<float>(2,d)/points4D.at<float>(3,d);
            points3D.emplace_back(point3D);
        }

        Mat rvec = Mat::eye(3, 3, CV_64F);;
        Mat tvec = (Mat_<double>(3, 1) << 0, 0, 0);
        vector<Point2f> imagePoints;
        projectPoints(points3D, rvec, tvec, M1, D1, imagePoints);

        Mat _rvec = R;
        Mat _tvec = T;
        vector<Point2f> _imagePoints;
        projectPoints(points3D, _rvec, _tvec, M2, D2, _imagePoints);

        for(int d=0; d<points3D.size(); d++)
        {
            // cout<<pointbuf[d]<<' '<<pointbuf1[d]<<' '<<imagePoints[d]<<' '<<_imagePoints[d]<<' '<<points3D[d]<<endl;
            cout<<_pointbuf[d]<<' '<<pointbuf2[d]<<' '<<_imagePoints[d]<<' '<<points3D[d]<<endl;
            circle(img1, imagePoints[d], 5, Scalar(20, 100, 100), 2);
            circle(img2, _imagePoints[d], 5, Scalar(20, 100, 100), 2);
        }
        imwrite("../save/left.jpg",img1);
        imwrite("../save/right.jpg",img2);

        waitKey();
    }
    #endif

    return 0;
}
