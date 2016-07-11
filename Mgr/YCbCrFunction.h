#pragma once
#include "stdafx.h"

namespace HandGR{
	/** Skin detection function working in YCbCr colour space. It also has some conditions from RGB colour space*/
	class YCbCrFunction: public SkinDetectionFunction{
		static const int range_cb_min = 110;
		static const int range_cb_max = 141;
		static const int range_cr_min = 128;//130
		static const int range_cr_max= 155;
	public:
		YCbCrFunction(bool initialFilterEnabled = true) : SkinDetectionFunction(initialFilterEnabled){}
		bool isSkin(float r, float g, float b);
		void gpuIsSkin(cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	private:
		inline float fitRange(float value, float low, float high);

	};
}