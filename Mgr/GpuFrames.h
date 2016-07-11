#include "stdafx.h"

namespace HandGR{
	struct GpuFrames{ //for CUDA Data allocations are very expensive on GPU. Use a buffer to solve: allocate once reuse later.
		cv::gpu::GpuMat frame, afterProcessingImage, afterSkinDetection, grayFrame, buf;
		std::vector<cv::gpu::GpuMat> chanels;
	};

}