#pragma once
#include "stdafx.h"
//#include <cxcore.hpp>
using namespace cv;

namespace HandGR{
	class HandFinder{

		bool saving;
		bool save;
		bool stop;

		int type;


		/** temp variable to moments */
		//Moments moments;
		double huMoments[7];

		vector<vector<cv::Point> > contours;

		//For position finder
		PositionDetector positionDetector;

		int prevRoiX;
		int prevRoiY;
		int prevSize;





		clock_t lastFrameTime;
		long frameDuration;

		GestureFitter gestureFitter;
		friend Teacher;
	public:
		HandFinder();
	private:

		/** calc distance between current huMoments and mean trained huMoments */
		double calcDistance(double* huMoments, const Mat& mean);

		void trainClassifiers();

		int trainOneGest( const CvMat &trSetSmall, int sampleCount, const int k, cv::Mat &trSet, cv::Mat &responses, const cv::Mat &meanMultiplier );

		void processKey( const int pressedKey );

		
		static void correctHandArea( vector<vector<cv::Point> >& contours);
		static void drawHandArea( vector<vector<Point> > &contours, Mat& imageWithMarks );

		bool findBestHandArea(FaceDetector& faceDetector, int &minCls, int &winnerConturId, Moments& winerMoments, double winnerHuMoments[7], vector<vector<Point> >& contours  );
		//static bool fitGesture(Mat& imageWithMarks, const Moments &winerMom, const double winnerHuMoments[7], int toFind, int winnerConturId, bool isCalibration, const vector<vector<Point> >& contours );
		bool mapResultToMouse( int bayesClassifierResult, int nearest1ClassifierResult, int nearest3ClassifierResult,  int svmClassifierARResult, int svmClassifierRKResult, int toFind );

	public:


		/** Find hand in picture, detect gest which is shown, and detect motion of hand*/
		bool findHand(Mat& imageAfterSkinRecognition, const Mat& frameImage, Mat& imageWithMarks, const int pressedKey, FaceDetector& faceDetector, bool isCalibration, int toFind = -1);
		bool gpuFindHand( const gpu::GpuMat afterSkinDetection, const gpu::GpuMat grayFrame, vector<gpu::GpuMat>& chanels, gpu::GpuMat buf,  Mat& imageWithMarks, const int pressedKey, FaceDetector& faceDetector, bool isCalibration, gpu::Stream& gpuStream, int toFind = -1 );
		

		

	};


}

