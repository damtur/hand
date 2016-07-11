#pragma once

#ifdef MYAPI_EXPORTS
#define MYAPI_API __declspec(dllexport)
#else
#define MYAPI_API __declspec(dllimport)
#endif

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/gpu/devmem2d.hpp>
#include <cuda_runtime.h>

#include "Pixel.h"

//struct MYAPI_API Pixel{
//	unsigned char b;
//	unsigned char g;
//	unsigned char r;
//};

__global__ void gpuEmptyFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst);
__global__ void gpuYCbCrFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst);
__global__ void gpuGausianFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst);
__global__ void gpuHsvFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst);
__global__ void gpuSimpleFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst);
__global__ void gpuSimpleFunction2(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst);
__global__ void gpuSimpleFunction3(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst);
__global__ void gpuSimpleFunction4(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst);
__global__ void gpuSimpleFunction5(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst);

__global__ void gpuInitialFilter(cv::gpu::DevMem2D_<Pixel> src);

enum FUNCTION{
	SIMPLE,
	SIMPLE2,
	SIMPLE3,
	SIMPLE4,
	SIMPLE5,
	YCBCR,
	EMPTY,
	HSV,
	GAUSSIAN
};

class MYAPI_API GpuSkinDetector{
public:
	static void yCbCrFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream);
	static void emptyFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream);
	static void gausianFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream);
	static void hsvFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	static void simpleFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	static void simpleFunction2(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	static void simpleFunction3(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	static void simpleFunction4(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	static void simpleFunction5(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	static void initialFilter( cv::gpu::DevMem2D_<Pixel>& src, cudaStream_t gpuStream);

private:
	static void detectSkin(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, const FUNCTION fun, cudaStream_t gpuStream);
};

