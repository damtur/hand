#include "stdafx.h"


namespace HandGR{
	/** Not used by the program. It was write to test gaussian skin detection function*/
	class GausianSkinDetectionFunction: public SkinDetectionFunction{
	public:
		GausianSkinDetectionFunction(bool initialFilterEnabled = true) : SkinDetectionFunction(initialFilterEnabled){}
		bool isSkin(float r, float g, float b);
		void gpuIsSkin(cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream );
	private:
	

		CvPoint3D64f GetMeanValue(IplImage *img);

		CvPoint3D64f GetDispersion(IplImage *img);

	};
}


