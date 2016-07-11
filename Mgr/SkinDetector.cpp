#include "stdafx.h"
using namespace cv;

namespace HandGR{

	SkinDetector::SkinDetector(){
		hsvFunction = new HsvFunction(true);
		ycbcrFunction = new YCbCrFunction(true);
		hsvFunctionF = new HsvFunction(false);
		ycbcrFunctionF = new YCbCrFunction(false);
		simpleSkinDetectionFunction = new SimpleSkinDetectionFunction(true);
		simpleSkinDetectionFunction2 = new SimpleSkinDetectionFunction2(true);
		simpleSkinDetectionFunction3 = new SimpleSkinDetectionFunction3(true);
		simpleSkinDetectionFunction4 = new SimpleSkinDetectionFunction4(true);
		simpleSkinDetectionFunction5 = new SimpleSkinDetectionFunction5(true);
		emptyFunction = new EmptyFunction(true);
		actualFunction = hsvFunctionF;
	}

	SkinDetector::~SkinDetector(){
		delete hsvFunction;
		delete ycbcrFunction;
		delete simpleSkinDetectionFunction;
		delete simpleSkinDetectionFunction2;
		delete simpleSkinDetectionFunction3;
		delete simpleSkinDetectionFunction4;
		delete simpleSkinDetectionFunction5;
		delete emptyFunction;
		delete ycbcrFunctionF;
		delete hsvFunctionF;
	}

	void SkinDetector::setActualFunction(const int pressedKey){
		if(pressedKey >= '0' && pressedKey <= '9'){
			Utils::globalCounter = 0;
		}

		switch(pressedKey){
		case '1':
			actualFunction = hsvFunction;
			break;
		case '2':
			actualFunction = ycbcrFunction;
			break;
		case '3':
			actualFunction = simpleSkinDetectionFunction;
			break;
		case '4':
			actualFunction = simpleSkinDetectionFunction2;
			break;
		case '5':
			actualFunction = simpleSkinDetectionFunction3;
			break;
		case '6':
			actualFunction = simpleSkinDetectionFunction4;
			break;
		case '7':
			actualFunction = simpleSkinDetectionFunction5;
			break;
		case '8':
			actualFunction = hsvFunctionF;
			break;
		case '9':
			actualFunction = ycbcrFunctionF;
			break;
		case '0':
			actualFunction = emptyFunction;
		}
	}


	void SkinDetector::detectSkin(const Mat& before, Mat& after){
double time = (double)getTickCount();
		actualFunction->detectSkin(before, after);

	if(Utils::globalCounter>=MAX_TEST_ITERATIONS){

		Utils::globalSaveToFile();
	}else{
		float tempTime = (((double)getTickCount() - time)/getTickFrequency());
		Utils::globalToSave.at<float>(Utils::globalCounter) = tempTime;
		cout<<tempTime<<endl;
	}
	ostringstream ss1;
	ss1 << (Utils::globalFunction - '0') << "_" << Utils::globalCounter<<"_";
	//Utils::printTime(time, ss1.str());
	cout<<endl;



	}

	void SkinDetector::detectSkin( const gpu::GpuMat& afterProcessingImage, gpu::GpuMat& afterSkinDetection, gpu::Stream& gpuStream ){
		actualFunction->detectSkin(afterProcessingImage, afterSkinDetection, gpuStream);
	}


}