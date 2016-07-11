#pragma once
#pragma warning(disable : 4512) 
#include "stdafx.h"

namespace HandGR{
	using namespace cv;

	class Calibration{
		boolean isCalibrated;

		cv::VideoCapture capture;

		/**information about static background*/
		Mat calibrationFrame;

		static const int numberOfCalibrationImages = 36;
		static const int thesholdLevel = 40;//czasem 30
		const float calibrationImagesDivider;
		bool calibrationImagesFilled;
		int refreshBackground;
		int currentCalibrationImageIndex;

		//GPU
		//gpu::GpuMat gpuCalibrationImages[numberOfCalibrationImages];
		gpu::GpuMat gpuCalibrationImages[numberOfCalibrationImages];
		//gpu::GpuMat gpuDifferResult, gpuTempMat;
		gpu::GpuMat gpuDifferResult, gpuStaticTempMat;
		
		//CPU
		Mat calibrationImages[numberOfCalibrationImages] ;
		Mat differResult;
		Mat tempMat;

	public:
		Calibration();



		/**Process image from camera - delete (make black) pixels which belong to static background
		/return rest pixels without change*/
		void oldClearBackground( const Mat& frame, Mat& afterProcessing, const int pressedKey);

		void clearBackground(const Mat& frame, Mat& afterProcessingImage);
		void gpuClearBackground( const gpu::GpuMat frame, const gpu::GpuMat grayFrame, gpu::GpuMat afterProcessingImage, gpu::GpuMat buf1, gpu::Stream& gpuStream, const int pressedKey);

		/**Start calibration process*/
		void init(cv::VideoCapture& capture, SkinDetector& skinDetector, HandFinder& handFinder, FaceDetector& faceDetector);

	private:

		/**save calibration frame*/
		void calibrate(const Mat& frame);

		/**Calibrate one gesture. Get information from camera about one gest to chose best skin detection function */
		void calibrateGest(int &pressedKey, cv::VideoCapture& capture, SkinDetector& skinDetector, HandFinder& handFinder,  FaceDetector& faceDetector, int* functioneEaluation, int testedGest, IplImage* hw, CvFont& font, CvScalar& color );

		int getBestFunction(int* functioneEvaluation);

		int diffOneColor(int i, int j, int chanelNr, int color, int correction[3]);
		bool diff(const Mat& mat, int i, int j, int correction[3]);
	};
}
