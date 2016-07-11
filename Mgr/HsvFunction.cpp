#include "stdafx.h"


namespace HandGR{
	bool HsvFunction::isSkin(float r, float g, float b){

		/* convert to HSV */
		float r_float;
		float g_float;
		float b_float;

		float h;
		float s;
		float v;

		r_float = ((float)r)/255.0f;
		g_float = ((float)g)/255.0f;
		b_float = ((float)b)/255.0f;

		v = MAX(r,MAX(g,b));
		s = (v == 0 ? 0 : ((v - MIN(r,MIN(g,b))) / v));

		if (v == r_float){
			h =       (g_float - b_float) * 60 / s;
		} else if (v == g_float) {
			h = 180 + (b_float - r_float) * 60 / s;
		} else { 
			h = 240 + (r_float - g_float) * 60 / s;
		}

		if (h < 0){
			h += 360;
		}


		//Debug::message("r :%d, g: %d, b: %d", Debug::LEVEL_DEBUG, r, g, b);
		//Debug::message("(1) h: %f, s: %f, v: %f", Debug::LEVEL_DEBUG, h, s, v);

		s *= 255.0;
		v *= 255.0;
		/* end convert */

		int range_h_min;
		int range_h_max;
		int range_s_min;
		int range_s_max;


		range_h_min = 244;
		range_h_max = 267;
		range_s_min = 87;
		range_s_max = 200;

		//Debug::log("(2) h: %f, s: %f, v: %f |||| r: %d, g: %d, b: %d", Debug::LEVEL_DEBUG, h, s, v, r, g, b);
		float sum = r+b+g;
		return (        
			h >= range_h_min && 
			h <= range_h_max &&      
			s >= range_s_min && 
			s <= range_s_max
			&&r>95&&g> 40 && b > 20 
			&& (b/g<1.249)&& b/g > 0.5 &&
			(sum/(3*r)>tc/0.692)&&
			(0.3333-b/sum>0.029)&&
			(g/(3*sum)<0.124)

			||	(3*b*r*r)/(sum*sum*sum) >0.110 && 
			((r*b + g*g ) / g*b) > 5000 &&
			sum/(3*r + (r-g)/sum) <2.7775 


			);
	}

	void HsvFunction::gpuIsSkin( cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst,cudaStream_t gpuStream )
	{
		GpuSkinDetector::hsvFunction(src, dst, gpuStream);
	}


}