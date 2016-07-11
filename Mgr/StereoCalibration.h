#pragma once

#include "stdafx.h"

/*
void normalize(Mat& threeFrames, Mat& threNorm ) {
	float max = numeric_limits<float>::min();
	float min = numeric_limits<float>::max();
	for(int i = 0; i < threeFrames.rows; ++i){
		for(int j = 0; j < threeFrames.cols; ++j){
			float val = threeFrames.at<float>(i,j);
			if(-numeric_limits<float>::infinity() != val && numeric_limits<float>::infinity() != val && numeric_limits<float>::min() != val && numeric_limits<float>::max() != val){
				max = std::max(max, val);
				min = std::min(min, val);
			}
		}
	}

	float rangeBefore = max - min;
	float rangeAfter = 255;

	for(int i = 0; i < threeFrames.rows; ++i){
		for(int j = 0; j < threeFrames.cols; ++j){
			float var = threeFrames.at<float>(i,j);
			//cout<<var<<endl;
			if(var < max && var > min){
				threNorm.at<unsigned char>(i,j) = unsigned char(((var + 0 - min)/rangeBefore)*rangeAfter);
			}else{
				if(var>max){
					threNorm.at<unsigned char>(i,j) = 255;
				}else{
					threNorm.at<unsigned char>(i,j) = 0;
				}
			}
		}
	}
}/*/








//main


/*

//StereoCalibration stereoCalibration(Size(9,6), capture, capture2);

resize(frame, frame, Size(RESOLUTION_X, RESOLUTION_Y));
resize(frame2, frame2, Size(RESOLUTION_X, RESOLUTION_Y));

cvtColor(frame, frame, CV_BGR2GRAY);
cvtColor(frame2, frame2, CV_BGR2GRAY);


imshow("L", frame);
imshow("R", frame2);


stringstream str;
str<<"left"<< ((i < 10) ? "0" : "") << i <<".jpg";

stringstream str2;
str2<<"right"<< ((i < 10) ? "0" : "") << i <<".jpg";

imwrite(str.str(), frame);
imwrite(str2.str(), frame2);


if(i++ > 14) return 0;
cout<<"SAVE"<<endl;
continue;



stereoCalibration(frame, frame2, frame3d);

if(frame3d.data){
	Mat threeFrames[3];
	split(frame3d, threeFrames);

	Mat threNorm[3];

	//threNorm[0] = Mat(threeFrames[0].rows,threeFrames[0].cols, CV_8UC1);
	//threNorm[1] = Mat(threeFrames[0].rows,threeFrames[0].cols, CV_8UC1);
	threNorm[2] = Mat(threeFrames[0].rows,threeFrames[0].cols, CV_8UC1);

	//normalize(threeFrames[0], threNorm[0]);
	//normalize(threeFrames[1], threNorm[1]);
	normalize(threeFrames[2], threNorm[2]);


	//imshow("x", threNorm[0]);
	//imshow("y", threNorm[1]);
	imshow("z", threNorm[2]);
}



imshow("LEFT", frame);
imshow("RIGHT", frame2);


continue;

*/

using namespace cv;

namespace HandGR{

	class StereoCalibration{
	private:
		static const int calibrationIterations = 10;
		Mat Q;

		Mat disparity;

		StereoVar stereo;
		//StereoBM stereo;
		//StereoSGBM stereo;

	private:
	
		static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners){
			corners.resize(0);
    
			for( int i = 0; i < boardSize.height; i++ )
				for( int j = 0; j < boardSize.width; j++ )
					corners.push_back(Point3f(float(j*squareSize),
											  float(i*squareSize), 0));
		}

		static bool run2Calibration( vector<vector<Point2f> > imagePoints1, vector<vector<Point2f> > imagePoints2, Size imageSize, Size boardSize, float squareSize, 
									float aspectRatio, int flags, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Mat& R12, Mat& T12){
    
			// step 1: calibrate each camera individually
			vector<vector<Point3f> > objpt(1);
			vector<vector<Point2f> > imgpt;
			calcChessboardCorners(boardSize, squareSize, objpt[0]);
			vector<Mat> rvecs, tvecs;
    
			for(int c = 1; c <= 2; c++ ){
				const vector<vector<Point2f> >& imgpt0 = c == 1 ? imagePoints1 : imagePoints2;
				imgpt.clear();
				int N = 0;
				for(int i = 0; i < (int)imgpt0.size(); i++ ){
					if( !imgpt0[i].empty() ){
						imgpt.push_back(imgpt0[i]);
						N += (int)imgpt0[i].size();
					}
				}

				if( imgpt.size() < 3 ){
					printf("Error: not enough views for camera %d\n", c);
					return false;
				}

				objpt.resize(imgpt.size(),objpt[0]);
            
				Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
				if( flags & CV_CALIB_FIX_ASPECT_RATIO )
					cameraMatrix.at<double>(0,0) = aspectRatio;
        
				Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
        
				double err = calibrateCamera(objpt, imgpt, imageSize, cameraMatrix,
								distCoeffs, rvecs, tvecs,
								flags|CV_CALIB_FIX_K3/*|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_FIX_K6*/);
				bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
				if(!ok){
					printf("Error: camera %d was not calibrated\n", c);
					return false;
				}
				printf("Camera %d calibration reprojection error = %g\n", c, sqrt(err/N));
        
				if( c == 1 )
					cameraMatrix1 = cameraMatrix, distCoeffs1 = distCoeffs;
				else
					cameraMatrix2 = cameraMatrix, distCoeffs2 = distCoeffs;
			}
    
			vector<vector<Point2f> > imgpt_right;
    
			// step 2: calibrate (1,2)
			//const vector<vector<Point2f> >& imgpt0 = c == 2 ? imagePoints2 : imagePoints3;
        
			imgpt.clear();
			imgpt_right.clear();
			int N = 0;
        
			for(int i = 0; i < (int)std::min(imagePoints1.size(), imagePoints2.size()); i++ )
				if( !imagePoints1[i].empty() && !imagePoints2[i].empty() ){
					imgpt.push_back(imagePoints1[i]);
					imgpt_right.push_back(imagePoints2[i]);
					N += (int)imagePoints2[i].size();
				}
        
			if( imgpt.size() < 3 ){
				printf("Error: not enough shared views for cameras 1 and camera 2");
				return false;
			}
        
			objpt.resize(imgpt.size(),objpt[0]);
			Mat E, F;
			double err = stereoCalibrate(objpt, imgpt, imgpt_right, cameraMatrix1, distCoeffs1,
											cameraMatrix2, distCoeffs2,
											imageSize, R12, T12, E, F,
											TermCriteria(TermCriteria::COUNT, 30, 0),
											CV_CALIB_FIX_INTRINSIC);
			printf("Pair (1,2) calibration reprojection error = %g\n", sqrt(err/(N*2)));
			
			return true;
		}

		static void findChessboard(Mat& viewGray, const Size& boardSize, vector<vector<Point2f> > * imgpt, int k, int i ) {
			if(viewGray.data){
				vector<Point2f> ptvec;
				bool found = findChessboardCorners( viewGray, boardSize, ptvec, CV_CALIB_CB_ADAPTIVE_THRESH );

				drawChessboardCorners( viewGray, boardSize, Mat(ptvec), found );
				if( found ){
					imgpt[k][i].resize(ptvec.size());
					std::copy(ptvec.begin(), ptvec.end(), imgpt[k][i].begin());
				}
			}
		}

	public:
		StereoCalibration(const Size& boardSize, VideoCapture& captureLeft, VideoCapture& captureRight){
			int k;
			int flags = 0;//CV_CALIB_ZERO_TANGENT_DIST, CV_CALIB_FIX_PRINCIPAL_POINT
			Size imageSize(RESOLUTION_X, RESOLUTION_Y);
			float squareSize = 1.f, aspectRatio = 1.f;
    
			vector<vector<Point2f> > imgpt[2];

			Mat view, viewGray;
			Mat cameraMatrix[2], distCoeffs[2], R[2], P[2], R12, T12;
			for( k = 0; k < 2; ++k )
			{
				cameraMatrix[k] = Mat_<double>::eye(3,3);
				cameraMatrix[k].at<double>(0,0) = aspectRatio;
				cameraMatrix[k].at<double>(1,1) = 1;
				distCoeffs[k] = Mat_<double>::zeros(5,1);
				imgpt[k].resize(calibrationIterations);
			}
			Mat R13=Mat_<double>::eye(3,3), T13=Mat_<double>::zeros(3,1);
    
    
			

			Mat leftFrame, rightFrame;

			for(int currentCalibrationIteration = 0 ; currentCalibrationIteration < calibrationIterations ; ++currentCalibrationIteration){
				captureLeft >> leftFrame;
				captureRight >> rightFrame;

				resize(leftFrame, leftFrame, imageSize);
				resize(rightFrame, rightFrame, imageSize);
				cvtColor(leftFrame, leftFrame, CV_BGR2GRAY);
				cvtColor(rightFrame, rightFrame, CV_BGR2GRAY);
                
				findChessboard(leftFrame, boardSize, imgpt, 0, currentCalibrationIteration);
				findChessboard(rightFrame, boardSize, imgpt, 1, currentCalibrationIteration);
			}
    
			printf("Running 3d calibration ...\n");
    
			bool isOk = run2Calibration(imgpt[0], imgpt[1], imageSize, boardSize, squareSize, aspectRatio, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5,
							cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], R12, T12);
        
			// step 3: find rectification transforms
			if(isOk){
				stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R12, T12, R[0], R[1], P[0], P[1], Q);
			}
			
			
		}
		
		void operator()(const Mat& leftGrayFrame, const Mat& rightGrayFrame, Mat& frame3d){
			if(Q.data){
				stereo(rightGrayFrame, leftGrayFrame, disparity);
				
				Mat vdisp;
				//normalize(disparity, vdisp, 0, 256, CV_MINMAX );

				/*	for(int i = 0; i < disparity.rows; ++i){
				for(int j = 0; j < disparity.cols; ++j){
				short val = disparity.at<short>(i,j);
				if(-numeric_limits<short >::infinity() != val && numeric_limits<short>::infinity() != val && numeric_limits<short>::min() != val && numeric_limits<short>::max() != val){
				disparity.at<short>(i,j) = disparity.at<short>(i,j) / 16;
				}else{
				disparity.at<short>(i,j) = 0;
				}

				}
				}*/
				
				reprojectImageTo3D(disparity, frame3d, Q);
			}
		}
	};
}










/*
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

int print_help()
{
	cout <<
		" Given a list of chessboard images, the number of corners (nx, ny)\n"
			" on the chessboards, and a flag: useCalibrated for \n"
			"   calibrated (0) or\n"
			"   uncalibrated \n"
			"     (1: use cvStereoCalibrate(), 2: compute fundamental\n"
			"         matrix separately) stereo. \n"
			" Calibrate the cameras and display the\n"
			" rectified results along with the computed disparity images.	\n" << endl;
    cout << "Usage:\n ./stereo_calib -w board_width -h board_height [-nr ] <image list XML/YML file>\n" << endl;
    return 0;
}


static void
StereoCalib(const vector<string>& imagelist, Size boardSize, bool useCalibrated=true, bool showRectified=true)
{
    if( imagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }
    
    bool displayCorners = true;
    const int maxScale = 2;
    const float squareSize = 1.f;  // Set this to your actual square size
    // ARRAY AND VECTOR STORAGE:
    
    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;
    
    int i, j, k, nimages = (int)imagelist.size()/2;
    
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;
    
    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];
            Mat img = imread(filename, 0);
            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
                found = findChessboardCorners(timg, boardSize, corners, 
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, CV_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);
                imshow("corners", cimg1);
                char c = (char)waitKey(100);
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if( !found )
                break;
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                         TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                                      30, 0.01));
        }
        if( k == 2 )
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }
    
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);
    
    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(j*squareSize, k*squareSize, 0));
    }
    
    cout << "Running stereo calibration ...\n";
    
    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;
    
    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
                    CV_CALIB_FIX_ASPECT_RATIO +
                    CV_CALIB_ZERO_TANGENT_DIST +
                    CV_CALIB_SAME_FOCAL_LENGTH +
                    CV_CALIB_RATIONAL_MODEL +
                    CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
    cout << "done with RMS error=" << rms << endl;
    
// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average reprojection err = " <<  err/npoints << endl;
    
    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", CV_STORAGE_WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
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
        
    fs.open("extrinsics.yml", CV_STORAGE_WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";
    
    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    
// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;
    
    Mat rmap[2][2];
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
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
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nimages; i++ )
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
        
        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    
    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }
    
    for( i = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            Mat img = imread(goodImageList[i*2+k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
            cvtColor(rimg, cimg, CV_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
            if( useCalibrated )
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf)); 
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            }
        }
        
        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for( j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char)waitKey();
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
}

                   
static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}
                   
int main(int argc, char** argv)
{
    Size boardSize;
    string imagelistfn;
    bool showRectified = true;
    
    for( int i = 1; i < argc; i++ )
    {
        if( string(argv[i]) == "-w" )
        {
            if( sscanf(argv[++i], "%d", &boardSize.width) != 1 || boardSize.width <= 0 )
            {
                cout << "invalid board width" << endl;
                return print_help();
            }
        }
        else if( string(argv[i]) == "-h" )
        {
            if( sscanf(argv[++i], "%d", &boardSize.height) != 1 || boardSize.height <= 0 )
            {
                cout << "invalid board height" << endl;
                return print_help();
            }
        }
        else if( string(argv[i]) == "-nr" )
            showRectified = false;
        else if( string(argv[i]) == "--help" )
            return print_help();
        else if( argv[i][0] == '-' )
        {
            cout << "invalid option " << argv[i] << endl;
            return 0;
        }
        else
            imagelistfn = argv[i];
    }
    
    if( imagelistfn == "" )
    {
        imagelistfn = "stereo_calib.xml";
        boardSize = Size(9, 6);
    }
    else if( boardSize.width <= 0 || boardSize.height <= 0 )
    {
        cout << "if you specified XML file with chessboards, you should also specify the board width and height (-w and -h options)" << endl; 
        return 0;
    }
    
    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return print_help();
    }
    
    StereoCalib(imagelist, boardSize, false, showRectified);
    return 0;
}



//*/





