#include "stdafx.h"
#include "GausianSkinDetectionFunction.h"

namespace HandGR{
	bool isSkin(float r, float g, float b){
		float sum = r+g+b;
		//float pR = (1.0 / (2 * M_PI * dispersion.x / 255)) * exp((-0.5) * (r - dispersion.x) * (r - dispersion.x) / (256 * 256));
		return (g/b - r/g<=-0.0905)&&
			((g*sum)/(b*(r-g))>3.4857)&&
			((sum*sum*sum)/(3*g*r*r)<=7.397)&&
			(sum/(9*r)-0.333 > -0.0976);
	}

	CvPoint3D64f GausianSkinDetectionFunction::GetMeanValue(IplImage *img){
		CvPoint3D64f e;
		e.x = e.y = e.z = 0;

		for(int i = 0 ; i < img->height; ++i){
			for (int j = 0; j < img->width; ++j){
				float b = ((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 0];
				float g = ((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 1];
				float r = ((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 2];

				e.x += r;
				e.y += g;
				e.z += b;
			}
		}

		e.x /= (img->width * img->height);
		e.y /= (img->width * img->height);
		e.z /= (img->width * img->height);

		return e;
	}

	CvPoint3D64f GausianSkinDetectionFunction::GetDispersion(IplImage *img){
		CvPoint3D64f d;
		d.x = d.y = d.z = 0;
		CvPoint3D64f e = GetMeanValue(img);

		for(int i = 0 ; i < img->height; ++i){
			for (int j = 0; j < img->width; ++j){
				float b = ((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 0];
				float g = ((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 1];
				float r = ((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 2];

				float diffR = r - (float)e.x;
				float diffG = g - (float)e.y;
				float diffB = b - (float)e.z;

				d.x += diffR * diffR;
				d.y += diffG * diffG;
				d.z += diffB * diffB;
			}
		}

		d.x /= (img->width * img->height);
		d.y /= (img->width * img->height);
		d.z /= (img->width * img->height);

		d.x = sqrt(d.x);
		d.y = sqrt(d.y);
		d.z = sqrt(d.z);

		return d;
	}

	void GausianSkinDetectionFunction::gpuIsSkin( cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ){
		GpuSkinDetector::gausianFunction(src, dst, gpuStream);
	}


}


