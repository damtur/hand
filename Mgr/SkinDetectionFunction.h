#pragma once
#include "stdafx.h"

namespace HandGR{
	/** Parent class for all skin detection functions */
	class SkinDetectionFunction{
	public:
		/** Used only in test to set value of some parameters in program */
		float ta,tb,tc;
	protected:
		/** Tell if use initial filter */
		bool initialFilterEnabled;
		SkinDetectionFunction(bool initialFilterEnabled = true) : initialFilterEnabled(initialFilterEnabled){ta=0;tb=0;tc=0;}
	public:
		/** Abstract function isSkin must be overloaded by extended class. It gets one piksel and tell if it belong to skin or not.*/
		virtual bool isSkin(float r, float g, float b) = 0;
		virtual void gpuIsSkin(cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ) = 0;

		/** Detect skin pixels in picture */
		void detectSkin(const cv::Mat& before, cv::Mat& after, uchar skinColor = 255, uchar nonSkinColor = 0);
		void detectSkin(const gpu::GpuMat& afterProcessingImage, gpu::GpuMat& afterSkinDetection, gpu::Stream& gpuStream);
	private:
		/** Use initial filter to define is the pixel is skin or not*/
		bool inline isSkinInitialFilter(float r, float g, float b);
	};
}