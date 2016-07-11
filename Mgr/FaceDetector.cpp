#include "stdafx.h"

namespace HandGR{
	const string FaceDetector::file_name = "haarcascade_frontalface_alt_tree.xml";

	CascadeClassifier FaceDetector::classifier =  CascadeClassifier();
	gpu::CascadeClassifier_GPU FaceDetector::gpuClassifier = gpu::CascadeClassifier_GPU( );

	shared_ptr<vector<Rect> > FaceDetector::actualFaces = NULL; 

	gpu::GpuMat FaceDetector::gpuActualFaces;
	gpu::GpuMat FaceDetector::gpuTempFaces;
	unsigned FaceDetector::facesFound = 0;

	int FaceDetector::counter = 0;
	int FaceDetector::old = 0;
	bool FaceDetector::gpuEnabled = false;
	bool FaceDetector::gpuFacesDownloaded = false;
	Mat FaceDetector::facesHost;

	FaceDetector::FaceDetector(bool gpuEnabled = false){
		this->gpuEnabled = gpuEnabled;
		String file_name = "haarcascade_frontalface_default.xml";

		if(gpuEnabled){
			if(!gpuClassifier.load(file_name)){
				fprintf( stderr, "Blad: Brak pliku %s z zestawem uczacym...\n",  file_name.c_str() );
#ifndef _DEBUG
				getchar();
				exit(EXIT_FAILURE);
#endif
			}
		}else{
			if(!classifier.load(file_name)){
				fprintf( stderr, "Blad: Brak pliku %s z zestawem uczacym...\n",  file_name.c_str() );
#ifndef _DEBUG
				getchar();
				exit(EXIT_FAILURE);
#endif
			}
		}
	}

	void FaceDetector::gpuFindFaces(const gpu::GpuMat& frame){
		//gpu::cvtColor(frame, grayFrame, CV_BGR2GRAY);
		unsigned detectionsNumber = FaceDetector::gpuClassifier.detectMultiScale(frame, gpuTempFaces);

		if(detectionsNumber > 0){
			gpuActualFaces = gpuTempFaces;
			facesFound = detectionsNumber;
			gpuFacesDownloaded = false;
		}else{
			if(old > 5){
				gpuActualFaces = gpuTempFaces;
				old = 0;
				facesFound = 0;
				gpuFacesDownloaded = false;
			}else
				++old;
		}
		counter = 30;
	}

	void __cdecl FaceDetector::func(void *p){
		if (Params *params = reinterpret_cast<Params*>(p)){
			erode(params->image, params->image, Mat());

			vector<Rect> *faces = new vector<Rect>();
			shared_ptr<vector<Rect> > temp = shared_ptr<vector<Rect> >(faces);
			FaceDetector::classifier.detectMultiScale(params->image, *temp);

			if(!faces->empty()){
				actualFaces = temp;
			}else{
				if(old > 5){
					actualFaces = temp;
					old = 0;
				}else{
					++old;
				}
			}
			counter = 30;
		}
		_endthread();
	}


//////////////////////////////////////////////////////////////////////////
// PUBLIC
//////////////////////////////////////////////////////////////////////////

	void FaceDetector::findFaces(const Mat& image){
		if(counter == 0){
			//params;
			params.image = image.clone();
			_beginthread(func, 0, &params);
			--counter;
		}else{
			--counter;
		}
	}

	void FaceDetector::findFaces(const gpu::GpuMat& grayFrame){
		if(counter == 0){
			gpuFindFaces(grayFrame);
		}else{
			--counter;
		}
	}


	const bool FaceDetector::inFace(int x, int y){
		if(gpuEnabled){
			if(facesFound > 0 ){
				Rect* faces = getGpuFaces().ptr<Rect>();
				for(unsigned i = 0; i < facesFound; ++i){
					if(Utils::collisionTest(x, y, faces[i])){
						return true;
					}
				}
			}
		}else{
			shared_ptr<vector<Rect> > faces = this->getFaces();
			if(faces  != NULL){
				for(unsigned i = 0; i < faces ->size(); ++i){
					if(Utils::collisionTest(x, y, (*faces)[i])){
						return true;
					}
				}
			}
		}
		return false;
	}

	void FaceDetector::drawFaces(Mat image){
		if(gpuEnabled){
			
			// download only detected number of rectangles
			if(facesFound > 0 ){
				Rect* faces = getGpuFaces().ptr<Rect>();
				for(unsigned i = 0; i < facesFound; ++i){
					cv::rectangle(image, faces[i], Scalar(255,0,0));
				}
			}
		}else{
			shared_ptr<vector<Rect> > faces = this->getFaces();
			if(faces  != NULL){
				for(unsigned i = 0; i < faces ->size(); ++i){
					rectangle(image, (*faces )[i], CV_RGB(255,0,0));
				}
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////
	//PRIVATE
	//////////////////////////////////////////////////////////////////////////

	const shared_ptr<vector<Rect> > FaceDetector::getFaces(){
		return actualFaces;
	}

	Mat& FaceDetector::getGpuFaces(){
		if(gpuFacesDownloaded){
			return facesHost;
		}
		//download only detected number of rectangles
		gpuActualFaces.colRange(0, facesFound).download(facesHost);
		gpuFacesDownloaded = true;
		return facesHost;
	}

}