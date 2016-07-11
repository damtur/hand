#pragma once

#ifdef MYAPI_EXPORTS
#define MYAPI_API __declspec(dllexport)
#else
#define MYAPI_API __declspec(dllimport)
#endif

#define STRUCT_TYPE_RECT 1
#define STRUCT_TYPE_CIRCLE 0

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/gpu/devmem2d.hpp>
#include <cuda_runtime.h>

__global__ void gpuBinaryErrode(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations = 1, const int structType = STRUCT_TYPE_CIRCLE);
__global__ void gpuGreyscaleErrode(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations = 1, const int structType = STRUCT_TYPE_CIRCLE);
__global__ void gpuBinaryDilate(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations = 1, const int structType = STRUCT_TYPE_CIRCLE);
__global__ void gpuGreyscaleDilate(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations = 1, const int structType = STRUCT_TYPE_CIRCLE);

class MYAPI_API GpuFunctions{
public:
	static void binaryErrode(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, const int iterations = 1, cudaStream_t gpuStream = 0, const int structType = STRUCT_TYPE_CIRCLE);
	static void grayscaleErrode(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, const int iterations = 1, cudaStream_t gpuStream = 0, const int structType = STRUCT_TYPE_CIRCLE);
	static void binaryDilate(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, const int iterations = 1, cudaStream_t gpuStream = 0, const int structType = STRUCT_TYPE_CIRCLE);
	static void grayscaleDilate(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, const int iterations = 1, cudaStream_t gpuStream = 0, const int structType = STRUCT_TYPE_CIRCLE);

	static void binaryOpen(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, cv::gpu::DevMem2D_<unsigned char>& buffor, const int iterations = 1, cudaStream_t gpuStream = 0, const int structType = STRUCT_TYPE_CIRCLE);
	static void grayscaleOpen(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, cv::gpu::DevMem2D_<unsigned char>& buffor, const int iterations = 1, cudaStream_t gpuStream = 0, const int structType = STRUCT_TYPE_CIRCLE);
	static void binaryClose(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, cv::gpu::DevMem2D_<unsigned char>& buffor, const int iterations = 1, cudaStream_t gpuStream = 0, const int structType = STRUCT_TYPE_CIRCLE);
	static void grayscaleClose(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, cv::gpu::DevMem2D_<unsigned char>& buffor, const int iterations = 1, cudaStream_t gpuStream = 0, const int structType = STRUCT_TYPE_CIRCLE);
};

