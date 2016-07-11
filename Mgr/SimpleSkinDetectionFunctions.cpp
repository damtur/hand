#include "stdafx.h"


namespace HandGR{

	bool SimpleSkinDetectionFunction::isSkin( float r, float g, float b ){
		float sum = r+g+b;

		float a1 = r/b;
		float a2 = (r*b)/(sum*sum);
		float a3 = (r*g)/(sum*sum);

		return ( a1 > 1.185 && a2 > 0.107 && a3 > 0.112);
	}


	void SimpleSkinDetectionFunction::gpuIsSkin( cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst,cudaStream_t gpuStream ){
		GpuSkinDetector::simpleFunction(src, dst, gpuStream);
	}



	bool SimpleSkinDetectionFunction2::isSkin(float r, float g, float b){
		float sum = r+g+b;

		return (
			r > 95 && g > 40 && b > 20 
			&& MAX(MAX(r,g),b) - MIN(MIN(r,g),b) > 15 
			&& abs(r-g) > 15 
			&& r > g 
			&& r > b 
			&& (b/g < 1.249)
			&& b/g > 0.5 
			&& (sum/(3*r) > 0.692) 
			&& (0.3333-b/sum > 0.029)
			&& (g/(3*sum) < 0.124) || (3*b*r*r)/(sum*sum*sum) > 0.110 
			&& ((r*b + g*g ) / g*b) > 5000 
			&& sum/(3*r + (r-g)/sum) <2.7775 //2.7775 lub czasem 1.06
			);
	}

	void SimpleSkinDetectionFunction2::gpuIsSkin( cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ){
		GpuSkinDetector::simpleFunction2(src, dst, gpuStream);
	}


	bool SimpleSkinDetectionFunction3::isSkin(float r, float g, float b){
		float sum = r+g+b;

		return (
			g / b - r / b <= -0.0905) 
			&& (sum / (3*r) + (r- g) / sum <= 0.9498);
	}

	void SimpleSkinDetectionFunction3::gpuIsSkin( cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ){
		GpuSkinDetector::simpleFunction3(src, dst, gpuStream);
	}

	bool  SimpleSkinDetectionFunction4::isSkin(float r, float g, float b){
		float sum = r+g+b;

		return 
			(b/g < 1.249)
			&& (sum/(3*r) > 0.692)
			&& (0.3333-b/sum > 0.029)
			&& (g/(3*sum) < 0.124);
	}

	void SimpleSkinDetectionFunction4::gpuIsSkin( cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ){
		GpuSkinDetector::simpleFunction4(src, dst, gpuStream);
	}

	bool SimpleSkinDetectionFunction5::isSkin(float r, float g, float b){
		float sum = r+g+b;

		return 
			g/b - r/g<=-0.0905
			&& (g*sum)/(b*(r-g)) > 3.4857
			&& (sum*sum*sum)/(3*g*r*r) <= 7.397
			&& sum/(9*r)-0.333 > -0.0976;
	}

	void SimpleSkinDetectionFunction5::gpuIsSkin( cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ){
		GpuSkinDetector::simpleFunction5(src, dst, gpuStream);
	}

}