#pragma once
#include "stdafx.h"

using namespace cv;

namespace HandGR{
	class GestureFitter{

		/** Bayes classifier to recogition shown hand gesture*/
		cv::NormalBayesClassifier bayesClassifier;

		/** Bayes classifier to recogition shown hand gesture*/
		cv::KNearest nerestClassifier;

		/** Svm classifier to recogition shown hand gesture check betwen R and K gest*/
		CvSVM  svmClassiefierRK;

		/** Svm classifier to recogition shown hand gesture check betwen R and K gest*/
		CvSVM  svmClassiefierAR;

		/** Svm classifier to recogition shown hand gesture check betwen R and K gest*/
		CvSVM  svmClassiefierRO;

		cv::Mat trainingClassesResponses;

		gpu::BruteForceMatcher_GPU<L2<float> > gpuMacher;
		

		
		int ACTIV_MOMENTS_COUNT;
		
		
		/** Mean data from hand gestures moments*/
		Mat huMomentsMean[7];
		
		double* meansData[7];

		bool mouseOn;

		int leftCounter;
		int middleCounter;
		int rightCounter;

		PositionDetector& positionDetector;
		
	public:
		GestureFitter(int ACTIV_MOMENTS_COUNT, PositionDetector& positionDetector);
		GestureFitter::~GestureFitter();

		/** Setter for mouse on value */
		void setMouseOn(bool val) { mouseOn = val; }
		
		/** Getter for mouse on value */
		bool isMouseOn() const { return mouseOn; }

		/** Getter for hy moments mean */
		Mat (&getHuMomentsMean())[7] { return huMomentsMean; }

		/** Getter for active moments count */
		int getActiveMomentsCount() const { return ACTIV_MOMENTS_COUNT; }


		//Find the best mach from contours
		bool fitGesture(Mat& imageWithMarks, const Moments &winerMom, const double winnerHuMoments[7], int toFind, int winnerConturId, bool isCalibration, const vector<vector<Point> >& contours );

		/** Find the nearest gesture to gesture given in toFind parameter */
		bool findNearestGesture( const double * winnerHuMoments, int toFind );

		/** Find and fit the best hand-shape gesture in all countures made by GPU */
		bool gpuFitGesture(const gpu::GpuMat& grayFrame, Mat& imageWithMarks, const Moments &winerMom, const double winnerHuMoments[7], int toFind, int winnerConturId, bool isCalibration, const vector<vector<Point> >& contours );

		/** Find the nearest gesture from winner HuMomenst */
		bool gpuFindNearestGesture( const double * winnerHuMoments, int toFind );


	private:
		void trainClassifiers();
		int trainOneGest(const Mat &trSetSmall, int sampleCount, const int k, cv::Mat &trSet, cv::Mat &responses, const cv::Mat &meanMultiplier );
		bool mapResultToMouse( int bayesClassifierResult, int nearest1ClassifierResult, int nearest3ClassifierResult, int svmClassifierARResult, int svmClassifierRKResult, int toFind );
		bool mapResultToMouse(int nearestClassifierResult, int knnClassifierresult, int toFind);

		bool handleUnknown( int toFind );
		bool handleR( int toFind );
		bool handleA( int toFind );
		bool handleT( int toFind );

		static string getStringFor(int i);
		static void printResult( int bayesClassifierResult, int nearest1ClassifierResult, int nearest3ClassifierResult, int svmClassifierARResult, int svmClassifierRKResult );
		void trainGpuMacher( Mat trSet );
	};

}

