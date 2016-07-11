#include "stdafx.h"

/** Simple skin detection function in RGB colour space */
namespace HandGR{
	class SimpleSkinDetectionFunction5: public SkinDetectionFunction{
	public:
		SimpleSkinDetectionFunction5(bool initialFilterEnabled = true) : SkinDetectionFunction(initialFilterEnabled){}
		bool isSkin(float r, float g, float b);
		void gpuIsSkin(cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	};
}