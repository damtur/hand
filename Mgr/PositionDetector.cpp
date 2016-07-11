#include "stdafx.h"

namespace HandGR{


	PositionDetector::PositionDetector() : detector(maxPointsSize, 0.01, 0.0){
		prevPoints = vector<Point2f>(maxPointsSize);
		ptsInitialized = false;
		currentPointNumber = 0;
		gpuPyrLKOpticalFlow.winSize = Size(10,10);
		gpuPyrLKOpticalFlow.iters = 20;
		gpuPyrLKOpticalFlow.derivLambda = 0.3;
		gpuPyrLKOpticalFlow.useInitialFlow = false;
	}



	void PositionDetector::detectPosition( const Mat& currentImage, bool calibration, bool mouseOn, long frameDuration ){
		if(!ptsInitialized){
			return;
		}

		Mat currentGrayImg;
		cvtColor(currentImage, currentGrayImg, CV_BGR2GRAY);
		if(prevGrayImg.empty()){
			prevGrayImg = currentGrayImg;
		}

		calcOpticalFlowPyrLK(prevGrayImg, currentGrayImg, prevPoints, currentPoints,
			status, err, Size(10,10), 3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03));

		prevGrayImg = currentGrayImg;

		calculateShift(mouseOn, frameDuration);
	}

	void PositionDetector::calculateShift( bool mouseOn, long frameDuration ){
		double shiftX = 0;
		double shiftY = 0;
		int foundPointsCounter = 0;

		for(unsigned i = 0; i < status.size(); ++i){
			if(status[i] ==	1){
				shiftX += currentPoints[i].x - prevPoints[i].x;
				shiftY += currentPoints[i].y - prevPoints[i].y;
				foundPointsCounter++;
			}
		}

		double countedShiftX = shiftX / foundPointsCounter;
		double countedShiftY = shiftY / foundPointsCounter;
		int currentPointIndex = 0;

		int sumOfDifferences = 0;
		for(unsigned i = 0; i < currentPointNumber; ++i){
			if(status[i] ==	1){
				if((countedShiftX<0&&(currentPoints[i].x - prevPoints[i].x)>0) || (countedShiftX>0&& (currentPoints[i].x - prevPoints[i].x)<0) ||
					(countedShiftY<0&&(currentPoints[i].y - prevPoints[i].y)>0) || (countedShiftY>0&& (currentPoints[i].y - prevPoints[i].y)<0) 
					){
						++sumOfDifferences;
				}
				prevPoints[currentPointIndex++] = currentPoints[i];
			}
		}

		if(mouseOn && currentPointIndex>10 && sumOfDifferences/(double)currentPointIndex < 0.5){
			shiftX /= currentPointIndex;
			shiftY /= currentPointIndex;

			//low pass filtration
			if(CLOCKS_PER_SEC*sqrt(shiftX*shiftX + shiftY*shiftY)/frameDuration < 350){
				OutputServer::getInstance()->move(shiftX, shiftY);
			}
		}

		currentPointNumber = currentPointIndex;
	}


	void PositionDetector::renewPositionPoints(const int& xMean, const int& yMean, const int& size, bool isCalibration ){
		int x = xMean- size;
		int y = yMean - size;

		for(unsigned i = 0 ; i < currentPointNumber; ++i){
			if(!Utils::collisionTest(prevPoints[i], x, y+size, size*2, size)){
				prevPoints[i].x = static_cast<float>(rand() % size*2 + x);
				prevPoints[i].y = static_cast<float>(rand() % size + y + size);
			}
		}

		for(unsigned i = currentPointNumber; i < maxPointsSize; ++i){
			prevPoints[i].x = static_cast<float>(rand() % size*2 + x);
			prevPoints[i].y = static_cast<float>(rand() % size + y + size);
		}
		currentPointNumber=maxPointsSize;
		ptsInitialized = true;
	}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//										GPU IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void PositionDetector::renewPositionPoints(const gpu::GpuMat& grayFrame, int xMean, int yMean, int size){
		renewPositionPoints( xMean, yMean, size, false);
		upload(prevPoints, gpuPrevPoints);

		//if(mask.empty()){
		//	mask = Mat(grayFrame.rows, grayFrame.cols, CV_8UC1);
		//}

		//for(int i = 0; i < mask.rows; ++i){
		//	for(int j = 0; j < mask.cols; ++j){
		//		mask.at<uchar>(i,j) = (Utils::collisionTest(j, i, xMean, yMean+size, size*2, size)) ? 255 : 0;
		//	}
		//}

		//detector(grayFrame, gpuPrevPoints, gpu::GpuMat(mask));//TODO OPTIMIZE!! copy of matrix!

		//ptsInitialized = true;
	}

	void PositionDetector::download(const gpu::GpuMat& d_mat, vector<Point2f>& vec){
		if(!d_mat.empty()){
			vec.resize(d_mat.cols);
			Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
			d_mat.download(mat);
		}
	}

	void PositionDetector::upload(const vector<Point2f>& vec, gpu::GpuMat& d_mat){
		if(!vec.empty()){
			Mat mat(1, vec.size(), CV_32FC2, (void*)&vec[0]);
			d_mat.upload(mat);
		}
	}

	void PositionDetector::download(const gpu::GpuMat& d_mat, vector<uchar>& vec){
		if(!d_mat.empty()){
			vec.resize(d_mat.cols);
			Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
			d_mat.download(mat);
		}
	}

	void PositionDetector::detectPosition(const gpu::GpuMat grayFrame, bool calibration, bool mouseOn, long frameDuration){
		if(!ptsInitialized){
			return;
		}

		if(gpuPrevGrayImg.empty()){
			grayFrame.copyTo(gpuPrevGrayImg);
		}
		gpuPyrLKOpticalFlow.sparse(gpuPrevGrayImg, grayFrame, gpuPrevPoints, gpuCurrentPoints, gpuStatus);

		grayFrame.copyTo(gpuPrevGrayImg);

		download(gpuPrevPoints, prevPoints);
		download(gpuCurrentPoints, currentPoints);
		download(gpuStatus, status);

		currentPointNumber = status.size();
		calculateShift( mouseOn, frameDuration);
	}

	void PositionDetector::printPoints(Mat& imageWithMarks){
		for(unsigned i = 0; i < prevPoints.size(); ++i){
			circle(imageWithMarks, Point(static_cast<int>(prevPoints[i].x), static_cast<int>(prevPoints[i].y)), 2, Scalar(128,128,255));
		}
	}

}