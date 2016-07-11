#include "stdafx.h"

namespace HandGR{
	#pragma warning( disable: 4100)
	bool EmptyFunction::isSkin(float r, float g, float b){
		return true;
	}

	void EmptyFunction::gpuIsSkin( cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream )
	{
		GpuSkinDetector::emptyFunction(src, dst, gpuStream);
	}

#pragma warning( default: 4100 )
}