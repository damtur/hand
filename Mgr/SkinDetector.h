#include "stdafx.h"

#include "SkinDetectionFunction.h"
#include "HsvFunction.h"
#include "YCbCrFunction.h"
#include "SimpleSkinDetectionFunction.h"
#include "SimpleSkinDetectionFunction2.h"
#include "SimpleSkinDetectionFunction3.h"
#include "SimpleSkinDetectionFunction4.h"
#include "SimpleSkinDetectionFunction5.h"
#include "EmptyFunction.h"

using namespace cv;

namespace HandGR{
	/** Class which deals skin detection in the picture */
	class SkinDetector{
			SkinDetectionFunction* hsvFunction;
			SkinDetectionFunction* ycbcrFunction;
			SkinDetectionFunction* simpleSkinDetectionFunction;
			SkinDetectionFunction* simpleSkinDetectionFunction2;
			SkinDetectionFunction* simpleSkinDetectionFunction3;
			SkinDetectionFunction* simpleSkinDetectionFunction4;
			SkinDetectionFunction* simpleSkinDetectionFunction5;
			SkinDetectionFunction* emptyFunction;
			SkinDetectionFunction* hsvFunctionF;
			SkinDetectionFunction* ycbcrFunctionF;
			SkinDetectionFunction* actualFunction;


	public:
		SkinDetector();

		~SkinDetector();

		/** Change actual skin detection function 
			1 - hsv with initial filter
			2 - YCbCr with initial filter
			3-7 - RGB functions with initial filter
			8 - hsv without initial filter
			9 - YCbCr without initial filter
			0 - initial filter
		*/
		void setActualFunction(const int pressedKey);

		//* find skin in picture and make skin pixels white and background picture black */
		void detectSkin(const Mat& before, Mat& after);

		void detectSkin(const gpu::GpuMat& afterProcessingImage, gpu::GpuMat& afterSkinDetection, gpu::Stream& gpuStream);
	};
}