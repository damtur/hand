#include "stdafx.h"

using namespace HandGR;
using namespace cv;


int main(){
	cout.imbue(locale("french"));

	//Teacher::processImages("t", 500, "new_t.txt", 7);
	//return 0;
	
	Mat frame;
	Mat afterProcessingImage;
	Mat afterSkinDetection;
	Mat frameWithMarks;
	Mat readyToHandFind;

	GpuFrames gpuFrames;

	//Init video capture from camera
	VideoCapture capture = cv::VideoCapture(CV_CAP_ANY);

	capture.set(CV_CAP_PROP_FRAME_WIDTH, RESOLUTION_X);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, RESOLUTION_Y);

	SkinDetector skinDetector;
	HandFinder handFinder;

	OutputServer::getInstance();
	
#ifdef GPUON
	FaceDetector faceDetector = FaceDetector(true);
	cout<< "------------------GPU " << (gpu::getCudaEnabledDeviceCount() ? "Enabled" : "Disabled") << "----------------"<< endl;
	
#else
	FaceDetector faceDetector = FaceDetector(false);
	cout<< "------------------CPU-------------------"  << endl;
#endif

	if( !capture.retrieve(frame) ) {
		cerr << "Blad: brak zainstalowanej kamery..." << endl;
		getchar();
		return 1;
	}

	Calibration calibration;

	//Starts calibration process
	//calibration.init(capture, skinDetector, handFinder, faceDetector);

	bool saving = false;
	long saveCount = 0;
	int pressedKey;

	gpu::Stream gpuStream;

	double time;
	
	

	//Main program loop
	while( (pressedKey = cv::waitKey(10)) != 27 ) {
		
		//Keyboard handler
		if(pressedKey == 's')
			saving = saving ? false : true;
		else if (pressedKey >= '0' && pressedKey <= '9'){
			skinDetector.setActualFunction(pressedKey);
		}

		//Get next frame
		capture >> frame;	

		double overalTime = Utils::globalTime = time = (double)getTickCount();
	
	
		//Equalize color
		Utils::equalizeColor(frame);


//CPU//////////////////////////////////////////////////////////////////////////
#ifndef GPUON
		//Utils::equalizeImage(frame);

		//Prepere frames
		frameWithMarks = frame.clone();
		if(afterSkinDetection.empty()){
			afterSkinDetection = Mat(frame.size(), frame.type());
			afterProcessingImage = Mat(frame.size(), frame.type());
		}

		//Find faces
		faceDetector.findFaces(frame);

		//Static image filtering
		calibration.clearBackground(frame, afterProcessingImage);
		//afterProcessingImage = frame.clone();

		//Detect the skin in image
		skinDetector.detectSkin(afterProcessingImage, afterSkinDetection);


		//Find the hand and make decision
		bool handFound = handFinder.findHand(afterSkinDetection, frame, frameWithMarks, pressedKey, faceDetector, false);
	
#else
//GPU//////////////////////////////////////////////////////////////////////////
		
		frameWithMarks = frame.clone();

		gpuStream.enqueueUpload(frame,gpuFrames.frame);

		//Utils::equalizeImage(gpuFrames, gpuStream);

		//gpuStream.enqueueCopy(gpuFrames.frame, gpuFrames.frameWithMarks);

		if(gpuFrames.afterSkinDetection.empty()){
			gpu::split(gpuFrames.frame, gpuFrames.chanels, gpuStream);

			gpuStream.enqueueCopy(gpuFrames.frame, gpuFrames.afterSkinDetection);
			gpuStream.enqueueCopy(gpuFrames.frame, gpuFrames.afterProcessingImage);
			gpuStream.enqueueCopy(gpuFrames.chanels[0], gpuFrames.buf);
			afterSkinDetection = Mat(frame.size(), frame.type());
			afterProcessingImage = Mat(frame.size(), frame.type());
			readyToHandFind = Mat(frame.size(), CV_8UC1);
		}

		//Find the face
		gpu::cvtColor(gpuFrames.frame, gpuFrames.grayFrame, CV_BGR2GRAY, 0, gpuStream);
		faceDetector.findFaces(gpuFrames.grayFrame);

		//Clear the static background
		calibration.gpuClearBackground(gpuFrames.frame, gpuFrames.grayFrame, gpuFrames.afterProcessingImage, gpuFrames.buf, gpuStream, pressedKey);
		

		//Detect the skin in image
		skinDetector.detectSkin(gpuFrames.afterProcessingImage, gpuFrames.afterSkinDetection, gpuStream);

		//Find the hand
		bool handFound = handFinder.gpuFindHand(gpuFrames.afterSkinDetection, gpuFrames.grayFrame, gpuFrames.chanels, gpuFrames.buf, frameWithMarks, pressedKey, faceDetector, false, gpuStream);

#endif
//////////////////////////////////////////////////////////////////////////

		int fps = (int)(1/(((double)getTickCount() - overalTime)/getTickFrequency()));

#ifdef SHOW_IMAGES
		//Show images
#ifdef GPUON
		faceDetector.drawFaces(frameWithMarks);	
		rectangle(frameWithMarks, Point(0,0), Point(125,20), Scalar(0,0,0,0), CV_FILLED);
		putText(frameWithMarks, "Fps: " + Utils::toString(fps), Point(3,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255,0), 1);
		imshow("Obraz z kamery", frameWithMarks);


		//gpuStream.enqueueDownload(gpuFrames.afterSkinDetection, afterSkinDetection);
		//imshow("GPU afterSkinDetection",  afterSkinDetection);

		//gpuStream.enqueueDownload(gpuFrames.afterProcessingImage, afterProcessingImage);
		//imshow("GPU afterprocessingu",  afterProcessingImage);

		//gpuStream.enqueueDownload(gpuFrames.chanels[0], readyToHandFind);
		//imshow("GPU Ready to hand find",  readyToHandFind);

#else
		faceDetector.drawFaces(frameWithMarks);

		rectangle(frameWithMarks, Point(0,0), Point(125,20), Scalar(0,0,0,0), CV_FILLED);
		putText(frameWithMarks, "Fps: " + Utils::toString(fps), Point(3,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255,0), 1);
		imshow("Obraz z kamery", frameWithMarks);
		imshow("Po przetworzeniu filtrem wstępnym", afterProcessingImage);
		imshow("Po detekcji skory", afterSkinDetection);
#endif
#endif

		if(saving && handFound){
			Utils::saveImg(saveCount++, readyToHandFind);
		}
	}
	return 0;
}
//*/
