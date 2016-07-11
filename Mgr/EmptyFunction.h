#include "stdafx.h"

namespace HandGR{
	class EmptyFunction: public SkinDetectionFunction{
	public:
		/**  Useful only for if initial filter is on. All piksels are skin in this function.*/
		EmptyFunction(bool initialFilterEnabled = true) : SkinDetectionFunction(initialFilterEnabled){}
		bool isSkin(float r, float g, float b);
		void gpuIsSkin(cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	};
}