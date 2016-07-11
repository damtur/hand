#include "stdafx.h"

#include <process.h>

namespace HandGR{

	struct Params{
		Mat image;
	};

	class FaceDetector{
		static CascadeClassifier classifier;
		static shared_ptr<vector<Rect> > actualFaces; 
		static const string file_name;
		static int counter;
		static int old;
		Params params;
		static bool gpuEnabled;
		static void __cdecl func(void *p);

		static gpu::CascadeClassifier_GPU gpuClassifier;
		static gpu::GpuMat gpuActualFaces;
		static gpu::GpuMat gpuTempFaces;
		static unsigned facesFound;
		static bool gpuFacesDownloaded;
		static Mat facesHost;
	public:
		FaceDetector(bool gpuEnabled/* = false*/);

		/** Initiate face find. This function finds face only 1 to 30 invoking of this method. */
		void findFaces(const Mat& image);
		void findFaces(const gpu::GpuMat& grayFrame);

		/** Draw red rectangle in image where the faces are */
		void drawFaces(Mat image);

		/** Chceck if point is in face */
		const bool inFace(int x, int y);
	private:
		/** Get area where are the face found in last searching */
		void gpuFindFaces(const gpu::GpuMat& frame);

		const shared_ptr<vector<Rect> > getFaces();
		Mat& FaceDetector::getGpuFaces();
	};


}