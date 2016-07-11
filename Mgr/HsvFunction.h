#pragma once
#include "stdafx.h"


namespace HandGR{
	/** Skin detection function working in HSV colour space. It also has some conditions from RGB colour space*/
	class HsvFunction: public SkinDetectionFunction{
	public:
		HsvFunction(bool initialFilterEnabled = true) : SkinDetectionFunction(initialFilterEnabled){}
		bool isSkin(float r, float g, float b);
		void gpuIsSkin(cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst,cudaStream_t gpuStream );
	};
}