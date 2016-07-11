#pragma once
#include "stdafx.h"

using namespace cv;

namespace HandGR{
	class PositionDetector{

		Mat prevGrayImg;
		vector<Point2f> prevPoints;
		boolean ptsInitialized;
		vector<Point2f> currentPoints;

		/** Maximum number of points to follow in motion detection*/
		static const unsigned maxPointsSize = 100;

		unsigned currentPointNumber;

		vector<uchar> status;
		vector<float> err;

		/// GPU implementation
		gpu::PyrLKOpticalFlow gpuPyrLKOpticalFlow;
		gpu::GpuMat gpuPrevGrayImg;
		gpu::GpuMat gpuPrevPoints;
		gpu::GpuMat gpuCurrentPoints;
		gpu::GpuMat gpuTempNewPoints;
		gpu::GpuMat gpuStatus;

		gpu::GoodFeaturesToTrackDetector_GPU detector;
		Mat mask;


	public:
		PositionDetector();

		void detectPosition(const Mat& currentImage, bool calibration, bool mouseOn, long frameDuration);
		void detectPosition(const gpu::GpuMat grayFrame, bool calibration, bool mouseOn, long frameDuration);

		void renewPositionPoints(const int& xMean, const int& yMean, const int& size, bool calibration);
		void renewPositionPoints(const gpu::GpuMat& grayFrame, int xMean, int yMean, int size);

		void printPoints(Mat& imageWithMarks);
	private:
		void download(const gpu::GpuMat& d_mat, vector<Point2f>& vec);
		void upload(const vector<Point2f>& vec, gpu::GpuMat& d_mat);

		void download(const gpu::GpuMat& d_mat, vector<uchar>& vec);

		void calculateShift( bool mouseOn, long frameDuration );
	};
}

