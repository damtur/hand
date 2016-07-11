#include "stdafx.h"

namespace HandGR{

	bool YCbCrFunction::isSkin(float r, float g, float b){
		//return true;
		// convert to YCbCr 
		//y  =  0.2989f*r + 0.5866f*g + 0.1145f*b + 0.5f;
		float cb = -0.1687f*r - 0.3312f*g + 0.500f*b  + 128.0f;
		float cr =  0.500f*r  - 0.4183f*g - 0.0816f*b + 128.0f;

		//ImageUtils::fitRange(y , 0.0, 255.0);
		fitRange(cb, 0.0, 255.0);
		fitRange(cr, 0.0, 255.0);
		// end convert 

		float sum = r+g+b;

		//float a1 = r/b;
		//float a2 = (r*b)/(sum*sum);
		//float a3 = (r*g)/(sum*sum);
		return (

			r>95&&g> 40 && b > 20 
			&& MAX(MAX(r,g),b)-MIN(MIN(r,g),b) > 15 
			&& abs(r-g) > 15 && r > g && r > b 
			&& (b/g<1.249)&& b/g > 0.5 &&
			(sum/(3*r)>0.692)&&//0.696
			(0.3333-b/sum>0.029)&&//0.014
			(g/(3*sum)<0.124)

			||	(3*b*r*r)/(sum*sum*sum) >0.110  //0.1276
			&& ((r*b + g*g ) / g*b) > 5000 
			&& sum/(3*r + (r-g)/sum) < 2.7775 //2.7775 lub czasem 1.06
			&& cb >= range_cb_min 
			&& cb <= range_cb_max
			&& cr >= range_cr_min 
			&& cr <= range_cr_max
			);
	}

	inline float YCbCrFunction::fitRange(float value, float low, float high){
		return MIN(high, MAX(low, value));
	}

	void YCbCrFunction::gpuIsSkin( cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ){
		GpuSkinDetector::yCbCrFunction(src, dst, gpuStream);
	}


}