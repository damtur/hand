#pragma once
#include "stdafx.h"

namespace HandGR{
	/** Simple skin detection function in RGB colour space */
	class SimpleSkinDetectionFunction2: public SkinDetectionFunction{
	public:
		SimpleSkinDetectionFunction2(bool initialFilterEnabled = true) : SkinDetectionFunction(initialFilterEnabled){}
		bool isSkin(float r, float g, float b);
		void gpuIsSkin(cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	};
}		