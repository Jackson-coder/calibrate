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
#define pFLAG 2

static void saveXYZ(const char *filename, const Mat &mat)
{
    const double max_z = 1.0e4;
    FILE *fp = fopen(filename, "wt");
    for (int y = 0; y < mat.rows; y++)
    {
        for (int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
                continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
            //printf("%d %d\n",y,x);
        }
    }
    fclose(fp);
}

struct createhash
{
    size_t operator()(const cv::Point2f p) const
    {
        return hash<int>()(p.x) ^ hash<int>()(p.x);
    }
};
struct isEqual
{
    bool operator()(const Point2f pt1, const Point2f pt2) const
    {
        return pt1.x == pt2.x && pt1.y == pt2.y;
    }
};

int print_help_calib(char **argv)
{
    cout << " Given a list of chessboard images, the number of corners (nx, ny)\n"
            " on the chessboards, and a flag: useCalibrated for \n"
            "   calibrated (0) or\n"
            "   uncalibrated \n"
            "     (1: use stereoCalibrate(), 2: compute fundamental\n"
            "         matrix separately) stereo. \n"
            " Calibrate the cameras and display the\n"
            " rectified results along with the computed disparity images.   \n"
         << endl;
    cout << "Usage:\n " << argv[0] << " -w=<board_width default=9> -h=<board_height default=6> -s=<square_size default=1.0> <image list XML/YML file default=stereo_calib.xml>\n"
         << endl;
    return 0;
}

void StereoCalib(const vector<string> &imagelist, Size boardSize, float squareSize, bool displayCorners, bool useCalibrated, bool showRectified)
{
    if (imagelist.size() % 2 != 0)
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    const int maxScale = 2;
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f>> imagePoints[2];
    vector<vector<Point3f>> objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size() / 2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

    for (i = j = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++)
        {
            const string &filename = imagelist[i * 2 + k];
            Mat img = imread(filename, 0);
            cout << filename << endl;
            if (img.empty())
                break;

            if (imageSize == Size())
                imageSize = img.size();
            else if (img.size() != imageSize)
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f> &corners = imagePoints[k][j];
            for (int scale = 1; scale <= maxScale; scale++)
            {
                Mat timg;
                if (scale == 1)
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);
                found = findChessboardCorners(timg, boardSize, corners,
                                              CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                if (found)
                {
                    if (scale > 1)
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1. / scale;
                    }
                    break;
                }
            }
            if (displayCorners)
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640. / MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);
                imshow("corners", cimg1);
                char c = (char)waitKey(500);
                if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if (!found)
                break;
            cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
                                      30, 0.01));
        }
        if (k == 2)
        {
            goodImageList.push_back(imagelist[i * 2]);
            goodImageList.push_back(imagelist[i * 2 + 1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if (nimages < 2)
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for (i = 0; i < nimages; i++)
    {
        for (j = 0; j < boardSize.height; j++)
            for (k = 0; k < boardSize.width; k++)
                objectPoints[i].push_back(Point3f(k * squareSize, j * squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                                 cameraMatrix[0], distCoeffs[0],
                                 cameraMatrix[1], distCoeffs[1],
                                 imageSize, R, T, E, F,
                                 CALIB_FIX_ASPECT_RATIO +
                                     CALIB_ZERO_TANGENT_DIST +
                                     CALIB_USE_INTRINSIC_GUESS +
                                     CALIB_SAME_FOCAL_LENGTH +
                                     CALIB_RATIONAL_MODEL +
                                     CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
    cout << "done with RMS error=" << rms << endl;

    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for (i = 0; i < nimages; i++)
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for (k = 0; k < 2; k++)
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
        }
        for (j = 0; j < npt; j++)
        {
            double errij = fabs(imagePoints[0][i][j].x * lines[1][j][0] +
                                imagePoints[0][i][j].y * lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x * lines[0][j][0] +
                                imagePoints[1][i][j].y * lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average epipolar err = " << err / npoints << endl;

    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] << "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

    // COMPUTE AND DISPLAY RECTIFICATION
    if (!showRectified)
        return;

    Mat rmap[2][2];
    // IF BY CALIBRATED (BOUGUET'S METHOD)
    if (useCalibrated)
    {
        // we already computed everything
    }
    // OR ELSE HARTLEY'S METHOD
    else
    // use intrinsic parameters of each camera, but
    // compute the rectification transformation directly
    // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for (k = 0; k < 2; k++)
        {
            for (i = 0; i < nimages; i++)
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv() * H1 * cameraMatrix[0];
        R2 = cameraMatrix[1].inv() * H2 * cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo)
    {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h, w * 2, CV_8UC3);
    }
    else
    {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h * 2, w, CV_8UC3);
    }

    for (i = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++)
        {
            Mat img = imread(goodImageList[i * 2 + k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w * k, 0, w, h)) : canvas(Rect(0, h * k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            if (useCalibrated)
            {
                Rect vroi(cvRound(validRoi[k].x * sf), cvRound(validRoi[k].y * sf),
                          cvRound(validRoi[k].width * sf), cvRound(validRoi[k].height * sf));
                rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
            }
        }

        if (!isVerticalStereo)
            for (j = 0; j < canvas.rows; j += 16)
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for (j = 0; j < canvas.cols; j += 16)
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char)waitKey();
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }
}

bool readStringList1(const string &filename, vector<string> &l)
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
        //l.push_back("/home/camera_calibrate/"+(string)*it);
        l.push_back("/home/camera_calibrate/" + (string)*it);
    return true;
}

const char *usage =
    " \nexample command line for calibration from a live feed.\n"
    "   calibration  -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe\n"
    " \n"
    " example command line for calibration from a list of stored images:\n"
    "   imagelist_creator image_list.xml *.png\n"
    "   calibration -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe image_list.xml\n"
    " where image_list.xml is the standard OpenCV XML/YAML\n"
    " use imagelist_creator to create the xml or yaml list\n"
    " file consisting of the list of strings, e.g.:\n"
    " \n"
    "<?xml version=\"1.0\"?>\n"
    "<opencv_storage>\n"
    "<images>\n"
    "view000.png\n"
    "view001.png\n"
    "<!-- view002.png -->\n"
    "view003.png\n"
    "view010.png\n"
    "one_extra_view.jpg\n"
    "</images>\n"
    "</opencv_storage>\n";

const char *liveCaptureHelp =
    "When the live video from camera is used as input, the following hot-keys may be used:\n"
    "  <ESC>, 'q' - quit the program\n"
    "  'g' - start capturing images\n"
    "  'u' - switch undistortion on/off\n";

static void fish_calib_help(char **argv)
{
    printf("This is a camera calibration sample.\n"
           "Usage: %s\n"
           "     -w=<board_width>         # the number of inner corners per one of board dimension\n"
           "     -h=<board_height>        # the number of inner corners per another board dimension\n"
           "     [-pt=<pattern>]          # the type of pattern: chessboard or circles' grid\n"
           "     [-n=<number_of_frames>]  # the number of frames to use for calibration\n"
           "                              # (if not specified, it will be set to the number\n"
           "                              #  of board views actually available)\n"
           "     [-d=<delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
           "                              # (used only for video capturing)\n"
           "     [-s=<squareSize>]       # square size in some user-defined units (1 by default)\n"
           "     [-o=<out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
           "     [-op]                    # write detected feature points\n"
           "     [-oe]                    # write extrinsic parameters\n"
           "     [-oo]                    # write refined 3D object points\n"
           "     [-zt]                    # assume zero tangential distortion\n"
           "     [-a=<aspectRatio>]      # fix aspect ratio (fx/fy)\n"
           "     [-p]                     # fix the principal point at the center\n"
           "     [-v]                     # flip the captured images around the horizontal axis\n"
           "     [-V]                     # use a video file, and not an image list, uses\n"
           "                              # [input_data] string for the video file name\n"
           "     [-su]                    # show undistorted images after calibration\n"
           "     [-ws=<number_of_pixel>]  # Half of search window for cornerSubPix (11 by default)\n"
           "     [-dt=<distance>]         # actual distance between top-left and top-right corners of\n"
           "                              # the calibration grid. If this parameter is specified, a more\n"
           "                              # accurate calibration method will be used which may be better\n"
           "                              # with inaccurate, roughly planar target.\n"
           "     [input_data]             # input data, one of the following:\n"
           "                              #  - text file with a list of the images of the board\n"
           "                              #    the text file can be generated with imagelist_creator\n"
           "                              #  - name of video file with a video of the board\n"
           "                              # if input_data not specified, a live view from the camera is used\n"
           "\n",
           argv[0]);
    printf("\n%s", usage);
    printf("\n%s", liveCaptureHelp);
}

enum
{
    DETECTION = 0,
    CAPTURING = 1,
    CALIBRATED = 2
};
enum Pattern
{
    CHESSBOARD,
    CIRCLES_GRID,
    ASYMMETRIC_CIRCLES_GRID
};

static double computeReprojectionErrors(
    const vector<vector<Point3f>> &objectPoints,
    const vector<vector<Point2f>> &imagePoints,
    const vector<Mat> &rvecs, const vector<Mat> &tvecs,
    const Mat &cameraMatrix, const Mat &distCoeffs,
    vector<float> &perViewErrors)
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int)objectPoints.size(); i++)
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f> &corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    switch (patternType)
    {
    case CHESSBOARD:
    case CIRCLES_GRID:
        for (int i = 0; i < boardSize.height; i++)
            for (int j = 0; j < boardSize.width; j++)
                corners.push_back(Point3f(float(j * squareSize),
                                          float(i * squareSize), 0));
        break;

    case ASYMMETRIC_CIRCLES_GRID:
        for (int i = 0; i < boardSize.height; i++)
            for (int j = 0; j < boardSize.width; j++)
                corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
                                          float(i * squareSize), 0));
        break;

    default:
        CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

static bool runCalibration(vector<vector<Point2f>> imagePoints,
                           Size imageSize, Size boardSize, Pattern patternType,
                           float squareSize, float aspectRatio,
                           float grid_width, bool release_object,
                           int flags, Mat &cameraMatrix, Mat &distCoeffs,
                           vector<Mat> &rvecs, vector<Mat> &tvecs,
                           vector<float> &reprojErrs,
                           vector<Point3f> &newObjPoints,
                           double &totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (flags & CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f>> objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);
    objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    double rms;
    int iFixedPoint = -1;
    if (release_object)
        iFixedPoint = boardSize.width - 1;
    rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
                            cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
                            flags | CALIB_FIX_K3 | CALIB_USE_LU);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    if (release_object)
    {
        cout << "New board corners: " << endl;
        cout << newObjPoints[0] << endl;
        cout << newObjPoints[boardSize.width - 1] << endl;
        cout << newObjPoints[boardSize.width * (boardSize.height - 1)] << endl;
        cout << newObjPoints.back() << endl;
    }

    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

static void saveCameraParams(const string &filename,
                             Size imageSize, Size boardSize,
                             float squareSize, float aspectRatio, int flags,
                             const Mat &cameraMatrix, const Mat &distCoeffs,
                             const vector<Mat> &rvecs, const vector<Mat> &tvecs,
                             const vector<float> &reprojErrs,
                             const vector<vector<Point2f>> &imagePoints,
                             const vector<Point3f> &newObjPoints,
                             double totalAvgErr)
{
    FileStorage fs(filename, FileStorage::WRITE);

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    if (!rvecs.empty() || !reprojErrs.empty())
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if (flags & CALIB_FIX_ASPECT_RATIO)
        fs << "aspectRatio" << aspectRatio;

    if (flags != 0)
    {
        sprintf(buf, "flags: %s%s%s%s",
                flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if (!reprojErrs.empty())
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if (!rvecs.empty() && !tvecs.empty())
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for (int i = 0; i < (int)rvecs.size(); i++)
        {
            Mat r = bigmat(Range(i, i + 1), Range(0, 3));
            Mat t = bigmat(Range(i, i + 1), Range(3, 6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if (!imagePoints.empty())
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for (int i = 0; i < (int)imagePoints.size(); i++)
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }

    if (!newObjPoints.empty())
    {
        fs << "grid_points" << newObjPoints;
    }
}

static bool readStringList2(const string &filename, vector<string> &l)
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    size_t dir_pos = filename.rfind('/');
    if (dir_pos == string::npos)
        dir_pos = filename.rfind('\\');
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
    {
        string fname = (string)*it;
        if (dir_pos != string::npos)
        {
            string fpath = samples::findFile(filename.substr(0, dir_pos + 1) + fname, false);
            if (fpath.empty())
            {
                fpath = samples::findFile(fname);
            }
            fname = fpath;
        }
        else
        {
            fname = samples::findFile(fname);
        }
        l.push_back(fname);
    }
    return true;
}

static bool runAndSave(const string &outputFilename,
                       const vector<vector<Point2f>> &imagePoints,
                       Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                       float grid_width, bool release_object,
                       float aspectRatio, int flags, Mat &cameraMatrix,
                       Mat &distCoeffs, bool writeExtrinsics, bool writePoints, bool writeGrid)
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    vector<Point3f> newObjPoints;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
                             aspectRatio, grid_width, release_object, flags, cameraMatrix, distCoeffs,
                             rvecs, tvecs, reprojErrs, newObjPoints, totalAvgErr);
    printf("%s. avg reprojection error = %.7f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);

    if (ok)
        saveCameraParams(outputFilename, imageSize,
                         boardSize, squareSize, aspectRatio,
                         flags, cameraMatrix, distCoeffs,
                         writeExtrinsics ? rvecs : vector<Mat>(),
                         writeExtrinsics ? tvecs : vector<Mat>(),
                         writeExtrinsics ? reprojErrs : vector<float>(),
                         writePoints ? imagePoints : vector<vector<Point2f>>(),
                         writeGrid ? newObjPoints : vector<Point3f>(),
                         totalAvgErr);
    return ok;
}

void print_help_match(char **argv)
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: %s <left_image> <right_image> [--algorithm=bm|sgbm|hh|hh4|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
           "[--no-display] [--color] [-o=<disparity_image>] [-p=<point_cloud_file>]\n",
           argv[0]);
}

static void print_help_convert(char **argv)
{
    printf("This is a camera calibration sample.\n"
           "Usage: %s\n"
           "     [image_data]             # one distorted image to convert \n"
           "     [input_data1]            # intrinsics parameters of fish model\n"
           "     [input_data2]            # intrinsics parameters of fictive rgb model\n"
           "\n",
           argv[0]);
}

// unordered_map<cv::Point2f,int,createhash,isEqual> a;

int main(int argc, char **argv)
{
#ifndef pFLAG
    cout << "NO DEFINITION OF PATTERN!!!" << endl;
#elif pFLAG == 0
    Size boardSize;
    string imagelistfn;
    bool showRectified;
    //cv::CommandLineParser parser(argc, argv, "{w|11|}{h|8|}{s|0.03|}{nr||}{help||}{@input|stereo_calib.xml|}");
    cv::CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{s|1.0|}{nr||}{help||}{@input|stereo_calib.xml|}");
    if (parser.has("help"))
        return print_help_calib(argv);
    showRectified = !parser.has("nr");
    imagelistfn = samples::findFile(parser.get<string>("@input"));
    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");
    float squareSize = parser.get<float>("s");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    vector<string> imagelist;
    bool ok = readStringList1(imagelistfn, imagelist);

    if (!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return print_help_calib(argv);
    }

    StereoCalib(imagelist, boardSize, squareSize, false, true, showRectified);

#elif pFLAG == 1

    Size boardSize, imageSize;
    float squareSize, aspectRatio = 1;
    Mat cameraMatrix, distCoeffs;
    string outputFilename;
    string inputFilename = "";

    int i, nframes;
    bool writeExtrinsics, writePoints;
    bool undistortImage = false;
    int flags = 0;
    VideoCapture capture;
    bool flipVertical;
    bool showUndistorted;
    bool videofile;
    int delay;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 0;
    vector<vector<Point2f>> imagePoints;
    vector<string> imageList;
    Pattern pattern = CHESSBOARD;

    cv::CommandLineParser parser(argc, argv,
                                 "{help ||}{w|11|}{h|8|}{pt|chessboard|}{n|10|}{d|1000|}{s|1|}{o|out_camera_data.yml|}"
                                 "{op||}{oe||}{zt||}{a||}{p||}{v||}{V||}{su||}"
                                 "{oo||}{ws|11|}{dt||}"
                                 "{@input_data|/home/camera_calibrate/pic_list.xml|}");
    if (parser.has("help"))
    {
        fish_calib_help(argv);
        return 0;
    }
    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");
    if (parser.has("pt"))
    {
        string val = parser.get<string>("pt");
        if (val == "circles")
            pattern = CIRCLES_GRID;
        else if (val == "acircles")
            pattern = ASYMMETRIC_CIRCLES_GRID;
        else if (val == "chessboard")
            pattern = CHESSBOARD;
        else
            return fprintf(stderr, "Invalid pattern type: must be chessboard or circles\n"), -1;
    }
    squareSize = parser.get<float>("s");
    nframes = parser.get<int>("n");
    delay = parser.get<int>("d");
    writePoints = parser.has("op");
    writeExtrinsics = parser.has("oe");
    bool writeGrid = parser.has("oo");
    if (parser.has("a"))
    {
        flags |= CALIB_FIX_ASPECT_RATIO;
        aspectRatio = parser.get<float>("a");
    }
    if (parser.has("zt"))
        flags |= CALIB_ZERO_TANGENT_DIST;
    if (parser.has("p"))
        flags |= CALIB_FIX_PRINCIPAL_POINT;
    flipVertical = parser.has("v");
    videofile = parser.has("V");
    if (parser.has("o"))
        outputFilename = parser.get<string>("o");
    showUndistorted = parser.has("su");
    if (isdigit(parser.get<string>("@input_data")[0]))
        cameraId = parser.get<int>("@input_data");
    else
        inputFilename = parser.get<string>("@input_data");
    int winSize = parser.get<int>("ws");
    float grid_width = squareSize * (boardSize.width - 1);
    bool release_object = false;
    if (parser.has("dt"))
    {
        grid_width = parser.get<float>("dt");
        release_object = true;
    }
    if (!parser.check())
    {
        fish_calib_help(argv);
        parser.printErrors();
        return -1;
    }
    if (squareSize <= 0)
        return fprintf(stderr, "Invalid board square width\n"), -1;
    if (nframes <= 3)
        return printf("Invalid number of images\n"), -1;
    if (aspectRatio <= 0)
        return printf("Invalid aspect ratio\n"), -1;
    if (delay <= 0)
        return printf("Invalid delay\n"), -1;
    if (boardSize.width <= 0)
        return fprintf(stderr, "Invalid board width\n"), -1;
    if (boardSize.height <= 0)
        return fprintf(stderr, "Invalid board height\n"), -1;

    if (!inputFilename.empty())
    {
        if (!videofile && readStringList2(samples::findFile(inputFilename), imageList))
            mode = CAPTURING;
        else
            capture.open(samples::findFileOrKeep(inputFilename));
    }
    else
        capture.open(cameraId);

    if (!capture.isOpened() && imageList.empty())
        return fprintf(stderr, "Could not initialize video (%d) capture\n", cameraId), -2;

    if (!imageList.empty())
        nframes = (int)imageList.size();

    if (capture.isOpened())
        printf("%s", liveCaptureHelp);

    namedWindow("Image View", 1);

    for (i = 0;; i++)
    {
        Mat view, viewGray;
        bool blink = false;

        if (capture.isOpened())
        {
            Mat view0;
            capture >> view0;
            view0.copyTo(view);
        }
        else if (i < (int)imageList.size())
            view = imread(imageList[i], 1);

        if (view.empty())
        {
            if (imagePoints.size() > 0)
                runAndSave(outputFilename, imagePoints, imageSize,
                           boardSize, pattern, squareSize, grid_width, release_object, aspectRatio,
                           flags, cameraMatrix, distCoeffs,
                           writeExtrinsics, writePoints, writeGrid);
            break;
        }

        imageSize = view.size();

        if (flipVertical)
            flip(view, view, 0);

        vector<Point2f> pointbuf;
        cvtColor(view, viewGray, COLOR_BGR2GRAY);

        bool found;
        switch (pattern)
        {
        case CHESSBOARD:
            found = findChessboardCorners(view, boardSize, pointbuf,
                                          CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
            break;
        case CIRCLES_GRID:
            found = findCirclesGrid(view, boardSize, pointbuf);
            break;
        case ASYMMETRIC_CIRCLES_GRID:
            found = findCirclesGrid(view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID);
            break;
        default:
            return fprintf(stderr, "Unknown pattern type\n"), -1;
        }

        // improve the found corners' coordinate accuracy
        if (pattern == CHESSBOARD && found)
            cornerSubPix(viewGray, pointbuf, Size(winSize, winSize),
                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

        if (mode == CAPTURING && found &&
            (!capture.isOpened() || clock() - prevTimestamp > delay * 1e-3 * CLOCKS_PER_SEC))
        {
            imagePoints.push_back(pointbuf);
            prevTimestamp = clock();
            blink = capture.isOpened();
        }

        if (found)
            drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

        string msg = mode == CAPTURING ? "100/100" : mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

        if (mode == CAPTURING)
        {
            if (undistortImage)
                msg = cv::format("%d/%d Undist", (int)imagePoints.size(), nframes);
            else
                msg = cv::format("%d/%d", (int)imagePoints.size(), nframes);
        }

        putText(view, msg, textOrigin, 1, 1,
                mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

        if (blink)
            bitwise_not(view, view);

        if (mode == CALIBRATED && undistortImage)
        {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);
        }

        imshow("Image View", view);
        char key = (char)waitKey(capture.isOpened() ? 50 : 500);

        if (key == 27)
            break;

        if (key == 'u' && mode == CALIBRATED)
            undistortImage = !undistortImage;

        if (capture.isOpened() && key == 'g')
        {
            mode = CAPTURING;
            imagePoints.clear();
        }

        if (mode == CAPTURING && imagePoints.size() >= (unsigned)nframes)
        {
            if (runAndSave(outputFilename, imagePoints, imageSize,
                           boardSize, pattern, squareSize, grid_width, release_object, aspectRatio,
                           flags, cameraMatrix, distCoeffs,
                           writeExtrinsics, writePoints, writeGrid))
                mode = CALIBRATED;
            else
                mode = DETECTION;
            if (!capture.isOpened())
                break;
        }
    }

    if (!capture.isOpened() && showUndistorted)
    {
        Mat view, rview, map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                                imageSize, CV_16SC2, map1, map2);

        for (i = 0; i < (int)imageList.size(); i++)
        {
            view = imread(imageList[i], 1);
            if (view.empty())
                continue;
            //undistort( view, rview, cameraMatrix, distCoeffs, cameraMatrix );
            remap(view, rview, map1, map2, INTER_LINEAR);
            imshow("Image View", rview);
            char c = (char)waitKey();
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    }

#elif pFLAG == 2

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
                                 "{@arg1|/home/camera_calibrate/left05.jpg|}{@arg2|/home/camera_calibrate/right05.jpg|}{w|9|}{h|6|}{ws|11|}{help h||}{algorithm|bm|}{max-disparity|256|}{blocksize|5|}{no-display||}{color||}{scale|1|}{i|/home/camera_calibrate/intrinsics.yml|}{e|/home/camera_calibrate/extrinsics.yml|}{o|record.jpg|}{p|record|}");
    if (parser.has("help"))
    {
        print_help_match(argv);
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
        print_help_match(argv);
        return -1;
    }
    if (numberOfDisparities < 1 || numberOfDisparities % 16 != 0)
    {
        printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
        print_help_match(argv);
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

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    Mat M1, D1, M2, D2;

    Mat R, T, R1, P1, R2, P2;

    //For single point, here is an example
    Mat map11, map12, map21, map22;

    int ccount = 0;
    unordered_map<cv::Point2f, Point2f, createhash, isEqual> pts;

    //Point2f inputPoint = Point2f(300, 400); //Point2f(312, 265);
    vector<Point2f> inputPoints;
    // for (int a = 0; a < 5; a++)
    // {
    //     inputPoints.emplace_back(Point2f(200 + a * 30, 300));
    // }

    Point2f outputPoint;
    vector<Point2f> pointbuf1, pointbuf2;
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

        //校正矩阵R1, R2;
        //世界坐标系下的投影矩阵P1, P2
        //R1是把校正前坐标系点坐标转换为校正后的点坐标所需的旋转矩阵
        stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);
        //fisheye::stereoRectify

        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12); //两个映射关系分别为x,y方向上的映射
        //fisheye::initUndistortRectifyMap
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
        //fisheye::initUndistortRectifyMap

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR); //双线性插值
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        //矫正图对应点测试
        for (auto inputPoint : inputPoints)
        {
            outputPoint.x = inputPoint.x + inputPoint.x - map11.at<Vec2C>(inputPoint.y, inputPoint.x)[0]; //不知道为啥
            outputPoint.y = inputPoint.y + inputPoint.y - map11.at<Vec2C>(inputPoint.y, inputPoint.x)[1]; //

            circle(img1, inputPoint, 10, Scalar(20, 100, 100), 2);
            circle(img1r, outputPoint, 10, Scalar(20, 100, 100), 2);
        }

        bool found1, found2;

        found1 = findChessboardCorners(img1, boardSize, pointbuf1,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        found2 = findChessboardCorners(img1r, boardSize, pointbuf2,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        // improve the found corners' coordinate accuracy
        if (found1)
        {
            cornerSubPix(img1, pointbuf1, Size(winSize, winSize),
                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
            drawChessboardCorners(img1, boardSize, Mat(pointbuf1), found1);
        }
        if (found2)
        {
            cornerSubPix(img1r, pointbuf2, Size(winSize, winSize),
                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
            drawChessboardCorners(img1r, boardSize, Mat(pointbuf2), found2);
        }


        img1.copyTo(image1);
        img2.copyTo(image2);

        imshow("畸变原图", img1);
        imshow("去畸变处理图", img1r);
        waitKey(0);

        img1 = img1r;
        img2 = img2r;
    }

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = img1.channels();

    sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
    sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    if (alg == STEREO_HH)
        sgbm->setMode(StereoSGBM::MODE_HH);
    else if (alg == STEREO_SGBM)
        sgbm->setMode(StereoSGBM::MODE_SGBM);
    else if (alg == STEREO_HH4)
        sgbm->setMode(StereoSGBM::MODE_HH4);
    else if (alg == STEREO_3WAY)
        sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

    Mat disp, disp8;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    float disparity_multiplier = 1.0f;
    if (alg == STEREO_BM)
    {
        bm->compute(img1, img2, disp);
        // cout << disp.at<int16_t>(477, 289) << endl;
        if (disp.type() == CV_16S)
            disparity_multiplier = 16.0f;
    }
    else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_HH4 || alg == STEREO_3WAY)
    {
        sgbm->compute(img1, img2, disp);
        if (disp.type() == CV_16S)
            disparity_multiplier = 16.0f;
    }
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if (alg != STEREO_VAR)
        disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities * 16.));
    else
        disp.convertTo(disp8, CV_8U);

    Mat disp8_3c;
    if (color_display)
        cv::applyColorMap(disp8, disp8_3c, COLORMAP_TURBO);

    if (!disparity_filename.empty())
        imwrite(disparity_filename, color_display ? disp8_3c : disp8);

    if (!point_cloud_filename.empty())
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;
        Mat floatDisp;
        disp.convertTo(floatDisp, CV_32F, 1.0f / disparity_multiplier);

        //根据Q[x y d 1]T = [X Y Z W]T可以求得任意

        reprojectImageTo3D(floatDisp, xyz, Q, true);
        //saveXYZ(point_cloud_filename.c_str(), xyz);

        Mat rvec = (Mat_<double>(3, 3) << P1.at<double>(0, 0), P1.at<double>(0, 1), P1.at<double>(0, 2),
                    P1.at<double>(1, 0), P1.at<double>(1, 1), P1.at<double>(1, 2),
                    P1.at<double>(2, 0), P1.at<double>(2, 1), P1.at<double>(2, 2));

        cv::Rodrigues(rvec, rvec);

        Mat tvec = (Mat_<double>(3, 1) << P1.at<double>(0, 3), P1.at<double>(1, 3), P1.at<double>(2, 3));

        const double max_z = 1.0e4;
        vector<vector<int>> error;

        //三维数据记录
        FILE *fp = fopen(point_cloud_filename.c_str(), "wt");

        vector<Point3f> objectPoints;
        vector<Point3f> output3dPoints;

        for (int y = 0; y < xyz.rows; y++)
        {
            vector<int> e;
            for (int x = 0; x < xyz.cols; x++)
            {
                e.push_back(0);
                Vec3f point = xyz.at<Vec3f>(y, x);
                if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
                    continue;
                fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);

                objectPoints.push_back(Point3f(point[0], point[1], point[2]));

                e.back() = 1;

                //printf("%d %d\n",y,x);
            }
            error.push_back(e);
        }
        fclose(fp);

        //测试点的三维信息
        for (auto poi : pointbuf2)
        {
            Vec3f point = xyz.at<Vec3f>(poi.y, poi.x);

            output3dPoints.push_back(Point3f(point[0], point[1], point[2]));
            cout << Point3f(point[0], point[1], point[2]) << endl;
        }

        //测试点的原始二维点
        // for (int hh = 0; hh < testoutputPoints2D.size(); hh++)
        // {
        //     bool flag = false;
        //     for (int j = 0; j < img1.rows; j++)
        //     {
        //         for (int i = 0; i < img1.cols; i++)
        //         {
        //             Point2f pt_in = Point2f(i, j);
        //             Point2f pt_out;
        //             pt_out.x = pt_in.x + pt_in.x - map11.at<Vec2C>(pt_in.y, pt_in.x)[0];
        //             pt_out.y = pt_in.y + pt_in.y - map11.at<Vec2C>(pt_in.y, pt_in.x)[1];

        //             if (pt_out == testoutputPoints2D[hh])
        //             {
        //                 testoriginPoints2D.push_back(pt_in);
        //                 flag = true;
        //                 break;
        //             }
        //         }
        //         if (flag)
        //             break;
        //     }
        //     if (!flag)
        //         testoutputPoints2D[hh] = Point2f(0, 0);
        // }

        //计算重投影误差
        vector<Point2f> imagePoints;

        // projectPoints(objectPoints, rvec, tvec, M1, D1, imagePoints);
        projectPoints(output3dPoints, rvec, tvec, M1, D1, imagePoints);
        //fisheye::projectPoints

        for (int tt = 0; tt < pointbuf1.size(); tt++)
        {
            circle(img1, pointbuf1[tt], 5, Scalar(20, 100, 100), 2);
            cout << pointbuf1[tt] << ' ' << pointbuf2[tt] << ' ' << imagePoints[tt] << ' '<< output3dPoints[tt] << ' '
                 << fabs(imagePoints[tt].y - pointbuf1[tt].y) + fabs(imagePoints[tt].x - pointbuf1[tt].x) << endl;
        }

        //printf("%d %d\n",y,x);
        printf("\n");

        //xyz是世界坐标系下的三维坐标集合，对应的是校正后img1中的点
        // cout << xyz.at<Vec3f>(outputPoint.x, outputPoint.y) << endl;
        // cout << xyz.at<Vec3f>(477, 289) << endl;
    }

    if (!no_display)
    {
        std::ostringstream oss;
        oss << "disparity  " << (alg == STEREO_BM ? "bm" : alg == STEREO_SGBM ? "sgbm" : alg == STEREO_HH ? "hh" : alg == STEREO_VAR ? "var" : alg == STEREO_HH4 ? "hh4" : alg == STEREO_3WAY ? "sgbm3way" : "");
        oss << "  blocksize:" << (alg == STEREO_BM ? SADWindowSize : sgbmWinSize);
        oss << "  max-disparity:" << numberOfDisparities;
        std::string disp_name = oss.str();

        namedWindow("oleft", cv::WINDOW_NORMAL);
        imshow("oleft", image1);
        namedWindow("left", cv::WINDOW_NORMAL);
        // circle(img1, Point(477, 289), 5, Scalar(20, 60, 50), 2);
        imshow("left", img1);
        namedWindow("right", cv::WINDOW_NORMAL);
        // circle(img2, Point(477, 289), 10, Scalar(20, 100, 100), 2);
        imshow("right", img2);
        namedWindow(disp_name, cv::WINDOW_AUTOSIZE);
        imshow(disp_name, color_display ? disp8_3c : disp8);

        printf("press ESC key or CTRL+C to close...");
        fflush(stdout);
        printf("\n");
        while (1)
        {
            if (waitKey() == 27) //ESC (prevents closing on actions like taking screenshots)
                break;
        }
    }

#elif pFLAG == 3

    cv::CommandLineParser parser(argc, argv, "{help h||}{@input|/home/fish_calibrate/001.jpg|}{@arg1|/home/fish_calibrate/out_camera_data.yml|}{@arg2|/home/fish_calibrate/convert_camera_data.yml|}");
    if (parser.has("help"))
    {
        print_help_convert(argv);
        return 0;
    }

    string image_file = samples::findFile(parser.get<std::string>(0));
    string input_data1 = samples::findFile(parser.get<std::string>(1));
    string input_data2 = samples::findFile(parser.get<std::string>(2));

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    FileStorage fs1(input_data1, FileStorage::READ);
    if (!fs1.isOpened())
    {
        printf("Failed to open file\n");
        return -1;
    }
    Mat camera_matrix1, distortion_coefficients1;
    fs1["camera_matrix"] >> camera_matrix1;
    fs1["distortion_coefficients"] >> distortion_coefficients1;

    // 畸变参数
    double k1 = distortion_coefficients1.at<double>(0, 0), k2 = distortion_coefficients1.at<double>(1, 0), p1 = distortion_coefficients1.at<double>(2, 0), p2 = distortion_coefficients1.at<double>(3, 0);
    // 内参
    double fx = camera_matrix1.at<double>(0, 0), fy = camera_matrix1.at<double>(1, 1), cx = camera_matrix1.at<double>(0, 2), cy = camera_matrix1.at<double>(1, 2);

    cv::Mat image = cv::imread(image_file, 0); // 图像是灰度图，CV_8UC1
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1); // 去畸变以后的图
    cv::Mat image_distort = cv::Mat(rows, cols, CV_8UC1);   //添加rgb畸变参数的图

    // 计算去畸变后图像的内容
    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
            double x = (u - cx) / fx, y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows)
            {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int)v_distorted, (int)u_distorted);
            }
            else
            {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }
    }

    FileStorage fs2(input_data2, FileStorage::READ);
    if (!fs2.isOpened())
    {
        printf("Failed to open file\n");
        return -1;
    }
    Mat camera_matrix2, distortion_coefficients2;
    fs2["camera_matrix"] >> camera_matrix2;
    fs2["distortion_coefficients"] >> distortion_coefficients2;

    // 畸变参数
    double _k1 = distortion_coefficients2.at<double>(0, 0), _k2 = distortion_coefficients2.at<double>(1, 0), _p1 = distortion_coefficients2.at<double>(2, 0), _p2 = distortion_coefficients2.at<double>(3, 0);
    // 内参
    double _fx = camera_matrix2.at<double>(0, 0), _fy = camera_matrix2.at<double>(1, 1), _cx = camera_matrix2.at<double>(0, 2), _cy = camera_matrix2.at<double>(1, 2);

    // 计算加畸变后图像的内容
    for (int v = 0; v < rows; v++)
    {
        for (int u = 0; u < cols; u++)
        {
            // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
            double x = (u - _cx) / _fx, y = (v - _cy) / _fy;
            double r = sqrt(x * x + y * y);
            double x_distort = x * (1 + _k1 * r * r + _k2 * r * r * r * r) + 2 * _p1 * x * y + _p2 * (r * r + 2 * x * x);
            double y_distort = y * (1 + _k1 * r * r + _k2 * r * r * r * r) + _p1 * (r * r + 2 * y * y) + 2 * _p2 * x * y;
            double u_distort = _fx * x_distort + _cx;
            double v_distort = _fy * y_distort + _cy;

            // 赋值 (最近邻插值)
            if (u_distort >= 0 && v_distort >= 0 && u_distort < cols && v_distort < rows)
            {
                image_distort.at<uchar>(v_distort, u_distort) = image_undistort.at<uchar>(v, u);
            }
        }
    }

    // 画图去畸变后图像
    namedWindow("distorted", cv::WINDOW_NORMAL);
    cv::imshow("distorted", image);
    namedWindow("undistorted", cv::WINDOW_NORMAL);
    cv::imshow("undistorted", image_undistort);
    namedWindow("image_distort", cv::WINDOW_NORMAL);
    cv::imshow("image_distort", image_distort);
    cv::waitKey();

#endif
    return 0;
}
