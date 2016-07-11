#include "stdafx.h"

namespace HandGR{


	/**save calibration frame*/
	void Calibration::calibrate(const Mat& frame){
		Mat frameToAcc;
		cout<< frame.type();
		Mat accumulator = Mat::zeros(frame.size(), CV_32FC3);
		for(int i = 0; i < 32; ++i){
			this->capture >> frameToAcc;

			Utils::equalizeImage(frameToAcc);

			//Morphology - to get rid off noise
			cv::erode(frameToAcc, frameToAcc, cv::Mat(), cv::Point(-1,-1), 1);
			cv::dilate(frameToAcc, frameToAcc, cv::Mat(), cv::Point(-1,-1), 1);

			//cout<<(int)frameToAcc.at<cv::Vec3b >(0,0)[0]<<endl;
			accumulate(frameToAcc, accumulator);
		}
		accumulator *= 1 / 32.0;
		accumulator.convertTo(calibrationFrame, frame.type());//frame.clone();
		isCalibrated = true;
		//imshow("Current background converted", calibrationFrame);
	}


	/**Calibrate one gest. Get information from camera about one gest to chose best skin detection function */
	void Calibration::calibrateGest(int &pressedKey, cv::VideoCapture& capture, SkinDetector& skinDetector, HandFinder& handFinder,  FaceDetector& faceDetector, int* functioneEaluation, int testedGest, IplImage* hw, CvFont& font, CvScalar& color ){
		Mat frame, afterSkinDetection, afterProcessingImage, frameWithMarks;
		int calibrationCounter = 0;
		bool calibrationStart = false;

		while( (pressedKey = cv::waitKey(10)) != 27 && calibrationCounter < 100){
			//Get next frame
			capture >> frame;
			if(afterSkinDetection.empty()){
				afterSkinDetection = Mat(frame.size(), frame.type());
				afterProcessingImage = Mat(frame.size(), frame.type());
			}
			imshow("Obraz z kamery", frame);

			if(pressedKey == ' ' && !calibrationStart){
				calibrationStart = true;
				cvSet(hw,cvScalar(0,0,0));

			}

			if(calibrationStart){
				stringstream str;
				str << "Czekaj..." << calibrationCounter << "%";
				cvSet(hw,cvScalar(0,0,0));
				cvPutText(hw, "Proces kalibracji", cvPoint( 10,30 ), &font, color);
				cvPutText(hw, str.str().c_str(), cvPoint( 10,70 ), &font, color);
				cvShowImage("Proces kalibracji", hw);
				++calibrationCounter;


				//Morphology - to get rid off noise
				cv::erode(frame, frame, cv::Mat(), cv::Point(-1,-1), 1);
				cv::dilate(frame, frame, cv::Mat(), cv::Point(-1,-1), 1);

				//Find the face
				faceDetector.findFaces(frame);

				//Static image filtering
				oldClearBackground(frame, afterProcessingImage, pressedKey);

				//Detect the skin in image
				for(int i = 0; i < 10 ; ++ i){ 
					skinDetector.setActualFunction('0' + i);

					skinDetector.detectSkin(afterProcessingImage, afterSkinDetection);
					//Morphology
					cv::erode(afterSkinDetection,afterSkinDetection, cv::Mat(), cv::Point(-1,-1), 1);//2);
					cv::dilate(afterSkinDetection,afterSkinDetection, cv::Mat(), cv::Point(-1,-1), 3);//7);
					cv::erode(afterSkinDetection,afterSkinDetection, cv::Mat(), cv::Point(-1,-1), 2);//5);
					//Find the hand and make decision
					if( handFinder.findHand(afterSkinDetection, frame, frameWithMarks, pressedKey, faceDetector , true, testedGest) ){
						++functioneEaluation[i];
					}
				}
			}
		}

	}

	int Calibration::getBestFunction(int* functioneEvaluation){
		int max = 0;
		int maxFunction = 0;

		for(int i = 0; i < 10; ++i){
			if(functioneEvaluation[i] > max){
				maxFunction = i;
				max = functioneEvaluation[i];
			}
		}
		return maxFunction;
	}


	Calibration::Calibration(): calibrationImagesDivider(0.0625){
		calibrationImagesFilled = false;
		isCalibrated = false;
		currentCalibrationImageIndex = 0;
		refreshBackground = 0;
	}


	void Calibration::gpuClearBackground(const gpu::GpuMat frame, const gpu::GpuMat grayFrame, gpu::GpuMat afterProcessingImage, gpu::GpuMat buf1, gpu::Stream& gpuStream, const int pressedKey){

		//Utils::globalTime = (double)getTickCount();

		if(!calibrationImagesFilled){
			if(gpuDifferResult.empty()){
				gpuDifferResult = gpu::GpuMat(grayFrame.rows, grayFrame.cols, grayFrame.type());
				gpuStaticTempMat = gpu::GpuMat(grayFrame.rows, grayFrame.cols, grayFrame.type());
			}

			gpuStream.enqueueCopy(grayFrame, gpuCalibrationImages[currentCalibrationImageIndex++]);

			if(currentCalibrationImageIndex == numberOfCalibrationImages){
				calibrationImagesFilled = true;
				currentCalibrationImageIndex = 0;
			}
		}else{
			for(int i = 0; i < numberOfCalibrationImages; ++i){
				gpu::absdiff(gpuCalibrationImages[i], grayFrame, gpuStaticTempMat, gpuStream);
				if(i > 0){
					gpu::addWeighted(gpuDifferResult, 1, gpuStaticTempMat, calibrationImagesDivider, 0, gpuDifferResult, -1, gpuStream);
				}else{
					gpu::addWeighted(gpuDifferResult, 0, gpuStaticTempMat, calibrationImagesDivider, 0, gpuDifferResult, -1, gpuStream);
				}
				
			}

			currentCalibrationImageIndex = (currentCalibrationImageIndex + 1) % numberOfCalibrationImages;

			gpu::threshold(gpuDifferResult, gpuDifferResult, thesholdLevel, 255, CV_THRESH_BINARY, gpuStream);
	
			Utils::binaryOpen(gpuDifferResult, gpuDifferResult, buf1, 1, gpuStream, STRUCT_TYPE_RECT);
			Utils::binaryClose(gpuDifferResult, gpuDifferResult, buf1, 15, gpuStream, STRUCT_TYPE_RECT);


			gpu::cvtColor(gpuDifferResult, afterProcessingImage, CV_GRAY2BGR, 0, gpuStream);

			//color inversion for background update
			gpu::bitwise_not(gpuDifferResult, gpuDifferResult, gpu::GpuMat(), gpuStream);

			//Clear the background
			gpu::bitwise_and(frame, afterProcessingImage, afterProcessingImage, gpu::GpuMat(), gpuStream);


			//UPDATE STATIC BACKGROUND 
			int backgroundPixels = gpu::countNonZero(gpuDifferResult);

			if(pressedKey == 'c' || refreshBackground == 0 && backgroundPixels < (gpuDifferResult.cols * gpuDifferResult.rows * 1 / 5)){
				refreshBackground = numberOfCalibrationImages;
			}

			if(refreshBackground > 0){
				gpuStream.enqueueCopy(grayFrame, gpuCalibrationImages[currentCalibrationImageIndex]);
				--refreshBackground;
			}else{
				grayFrame.copyTo(gpuCalibrationImages[currentCalibrationImageIndex], gpuDifferResult);
			}
		}


		//if(Utils::globalCounter>=MAX_TEST_ITERATIONS){
		//	ostringstream ss;
		//	ss << grayFrame.cols << "x" <<grayFrame.rows << "GPUBackgroundSegmentation.txt";
		//	Utils::globalSaveToFile(ss.str());

		//	Utils::globalCounter = 0;
		//	exit(0);
		//}else{
		//	Utils::globalToSave.at<float>(Utils::globalCounter) = (((double)getTickCount() - Utils::globalTime)/getTickFrequency());
		//	++Utils::globalCounter;

		//	cout<< "GPUBackgroundSegmentation TIME: " << (((double)getTickCount() - Utils::globalTime)/getTickFrequency()) << endl;
		//}
	}

	void Calibration::clearBackground( const Mat& frame, Mat& afterProcessingImage){

		//Utils::globalTime = (double)getTickCount();

		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);

		if(!calibrationImagesFilled){
			if(tempMat.empty()){
				differResult = Mat(grayFrame.rows, grayFrame.cols, grayFrame.type());
				tempMat = Mat(grayFrame.rows, grayFrame.cols, grayFrame.type());
			}

			calibrationImages[currentCalibrationImageIndex++] = grayFrame.clone();

			if(currentCalibrationImageIndex == numberOfCalibrationImages){
				calibrationImagesFilled = true;
				currentCalibrationImageIndex = 0;
			}
		}else{
			for(int i = 0; i < numberOfCalibrationImages; ++i){
				absdiff(calibrationImages[i], grayFrame, tempMat);
				if(i > 0){
					addWeighted(differResult, 1, tempMat, calibrationImagesDivider, 0, differResult);
				}else{
					addWeighted(differResult, 0, tempMat, calibrationImagesDivider, 0, differResult);
				}
			}

			currentCalibrationImageIndex = (currentCalibrationImageIndex + 1) % numberOfCalibrationImages;

			threshold(differResult, differResult, thesholdLevel, 255, CV_THRESH_BINARY);

			morphologyEx(differResult, differResult, MORPH_OPEN, Mat(), Point(-1,-1), 1);
			morphologyEx(differResult, differResult, MORPH_CLOSE, Mat(), Point(-1,-1), 15);

			cvtColor(differResult, afterProcessingImage, CV_GRAY2BGR);

			//color inversion for background update
			bitwise_not(differResult, differResult);

			//clear the background
			bitwise_and(frame, afterProcessingImage, afterProcessingImage);

			//update static backgorund
			int nonZeroPixels = countNonZero(differResult);

			if(refreshBackground == 0 && nonZeroPixels < (differResult.cols * differResult.rows * 1 / 5)){
				refreshBackground = numberOfCalibrationImages;
			}

			if(refreshBackground > 0){
				calibrationImages[currentCalibrationImageIndex] = grayFrame.clone();
				--refreshBackground;
			}else{
				grayFrame.copyTo(calibrationImages[currentCalibrationImageIndex], differResult);
			}
		}

		/*if(Utils::globalCounter>=MAX_TEST_ITERATIONS){
			ostringstream ss;
			ss << grayFrame.cols << "x" <<grayFrame.rows << "CPUBackgroundSegmentation.txt";
			Utils::globalSaveToFile(ss.str());

			Utils::globalCounter = 0;
			exit(0);
		}else{
			Utils::globalToSave.at<float>(Utils::globalCounter) = (((double)getTickCount() - Utils::globalTime)/getTickFrequency());
			++Utils::globalCounter;

			cout<< "CPUBackgroundSegmentation TIME: " << (((double)getTickCount() - Utils::globalTime)/getTickFrequency()) << endl;
		}*/
	}

	

	/**Process image from camera - delete (make black) pixels which belong to static background
	/return rest pixels without change*/
	void Calibration::oldClearBackground( const Mat& frame, Mat& afterProcessing, const int pressedKey )
	{
		if(pressedKey == 'c' || pressedKey == 'C' ){
			calibrate(frame);
		}

		if(!isCalibrated){
			afterProcessing = frame.clone();
			return;
		}

		double rCorrection=0, gCorrection=0, bCorrection=0;
		//int divideCounter=100;

		for(int i = 0 ; i < 10; ++i){
			for(int j = 0 ; j < frame.cols ; ++j){
				bCorrection += frame.at<cv::Vec3b >(i,j)[0] - calibrationFrame.at<cv::Vec3b >(i,j)[0];
				gCorrection += frame.at<cv::Vec3b >(i,j)[1] - calibrationFrame.at<cv::Vec3b >(i,j)[1];
				rCorrection += frame.at<cv::Vec3b >(i,j)[2] - calibrationFrame.at<cv::Vec3b >(i,j)[2];
			}
		}

		double rgb_multiplier = 1/ (frame.cols*10);

		rCorrection *= rgb_multiplier;
		gCorrection *= rgb_multiplier;
		bCorrection *= rgb_multiplier;

		int correction[3];
		correction[0] = (int)bCorrection;
		correction[1] = (int)gCorrection;
		correction[2] = (int)rCorrection;

		//Mat differenceMat;
		//vector<Mat> chanels;
		//split(frame, chanels);
		//for(int i = 0; i < 3 ; ++ i){
		//	chanels[i] += correction[i];
		//}
		//merge(chanels, differenceMat);
		
		//absdiff(calibrationFrame, differenceMat, differenceMat);
		//Mat afterThreshold;
		//cvtColor(differenceMat, afterThreshold, CV_BGR2GRAY);	
		//imshow("przed tr", afterThreshold);
		//threshold(afterThreshold, afterThreshold, 10, 1, THRESH_BINARY);

		
		//erode(afterThreshold, afterThreshold, Mat(), Point(-1,-1), 1);
		//dilate(afterThreshold, afterThreshold, Mat(), Point(-1,-1), 1);
		//imshow("Po tresholdzie", afterThreshold);
		//imshow("Uda sie?", differenceMat);
		//cvtColor(afterThreshold, afterThreshold, CV_GRAY2BGR);
		//afterProcessing = frame.mul(afterThreshold);
		//return;


		//HARD WAY :P
		for(int i = 0 ; i < frame.rows; ++i){
			for (int j = 0; j < frame.cols; ++j){
				if (i == 0 || i == frame.rows -1 || j == 0 || j == frame.cols ){
					afterProcessing.at<cv::Vec3b >(i,j)[0] = afterProcessing.at<cv::Vec3b >(i,j)[1] = afterProcessing.at<cv::Vec3b >(i,j)[2] = 0; 
					continue;
				}

				if(diff(frame,i,j, correction)){
					afterProcessing.at<cv::Vec3b >(i,j)[0] = afterProcessing.at<cv::Vec3b >(i,j)[1] = afterProcessing.at<cv::Vec3b >(i,j)[2] = 0; 
				}else{
					afterProcessing.at<cv::Vec3b >(i,j)[0] = frame.at<cv::Vec3b >(i,j)[0];
					afterProcessing.at<cv::Vec3b >(i,j)[1] = frame.at<cv::Vec3b >(i,j)[1];
					afterProcessing.at<cv::Vec3b >(i,j)[2] = frame.at<cv::Vec3b >(i,j)[2];
				}
			}
		}
	}

	bool Calibration::diff( const Mat& mat, int i, int j, int correction[3])
	{
		int b = mat.at<cv::Vec3b >(i,j)[0];
		int g = mat.at<cv::Vec3b >(i,j)[1];
		int r = mat.at<cv::Vec3b >(i,j)[2];

		int overalSum = 0;
		//int diffs[3][3];
		for(int k = -1; k <= 1; k++){
			for(int l = -1; l <= 1; l++){
				//diffs[k+1][l+1] = diffOneColor(i+k,j+k,2,r, correction) + diffOneColor(i+k,j+k,1,g, correction) + diffOneColor(i+k,j+k,0,b, correction);
				//overalSum += diffs[k+1][l+1];
				overalSum += diffOneColor(i+k,j+k,2,r, correction) + diffOneColor(i+k,j+k,1,g, correction) + diffOneColor(i+k,j+k,0,b, correction);
			}
		}
		//cout<<overalSum<<endl;
		
		//return ((diffOneColor(i,j,2,r, correction) < 15 && diffOneColor(i,j,1,g, correction) < 15 && diffOneColor(i,j,0,b, correction) < 15) || overalSum < 100);
		return (overalSum < 450);
	}

	int Calibration::diffOneColor( int i, int j, int chanelNr, int color, int correction[3] )
	{
		return abs( MAX(MIN(calibrationFrame.at<cv::Vec3b >(i,j)[chanelNr] + correction[chanelNr], 255),0) -color );
	}

	/**Start calibration process*/
	void Calibration::init(cv::VideoCapture& capture, SkinDetector& skinDetector, HandFinder& handFinder, FaceDetector& faceDetector){
		cvNamedWindow("Proces kalibracji", CV_WINDOW_AUTOSIZE);
		cv::Mat frame;
		cv::Mat afterSkinDetection;
		int pressedKey;

		IplImage* hw = cvCreateImage(cvSize(800, 400), 8, 3);
		cvSet(hw,cvScalar(0,0,0));
		CvFont font;
		cvInitFont( &font, FONT_HERSHEY_COMPLEX_SMALL , 1.0, 1.0);
		CvScalar color = CV_RGB(255, 255, 255);

		cvPutText(hw, "Proces kalibracji", cvPoint( 10,30 ), &font, color);
		cvPutText(hw, "Aby skalibrowac system do poprawnego dzialania", cvPoint( 10,70 ), &font, color);
		cvPutText(hw, "postepuj zgodnie z kolejnymi wskazowkami:", cvPoint( 10, 95 ), &font, color);
		cvPutText(hw, "(Jezeli chcesz pominac proces kalibracji wciscnij ESC)", cvPoint( 10, 120), &font, color);

		cvPutText(hw, "Etap pierwszy - Pobranie statycznego tla:", cvPoint( 10, 150 ), &font, color);
		cvPutText(hw, "Wyjdz z kadru kamery i upewnij sie ze wszystkie", cvPoint( 10, 180 ), &font, color);
		cvPutText(hw, "ruchome obiekty nie beda widoczne, a nastepnie", cvPoint( 10, 205 ), &font, color);
		cvPutText(hw, "nacisnij spacje.", cvPoint( 10, 230 ), &font, color);


		cvShowImage("Proces kalibracji", hw);

		this->capture = capture;

		//Calibration loop
		while( (pressedKey = cv::waitKey(10)) != 27){
			//Get next frame
			capture >> frame;
			//cvShowImage("Obraz z kamery", &(IplImage)frame);
			imshow("Obraz z kamery", frame);

			if(pressedKey == ' '){
				calibrate(frame);
				//processImage(frame, 'c');
				break;
			}
		}
		if(pressedKey == 27){ 
			cvDestroyWindow("Proces kalibracji");
			return;
		}

		int functioneEaluation [10];
		for(int i = 0 ; i < 10 ; ++i) functioneEaluation[i] = 0;

		cvNamedWindow("Podglad gestu", CV_WINDOW_AUTOSIZE);
		IplImage* preview = cvLoadImage("open.jpg");
		if(preview == NULL){
			fprintf( stderr, "WARNING: No file open.jpg\n" );
		}
		cvShowImage("Podglad gestu", preview);

		cvSet(hw,cvScalar(0,0,0));
		cvPutText(hw, "Proces kalibracji", cvPoint( 10,30 ), &font, color);
		cvPutText(hw, "Etap drugi - gest neutralny:", cvPoint( 10, 70 ), &font, color);
		cvPutText(hw, "Pokaz przed kamera otwarta reke (gest neutralny) obok ", cvPoint( 10, 100 ), &font, color);
		cvPutText(hw, "twarzy, tak by byla cala widoczna w kadrze kamery, ale", cvPoint( 10, 125 ), &font, color);
		cvPutText(hw, "nie znajdowala sie blisko krawedzi kadru, a nastepnie", cvPoint( 10, 150 ), &font, color);
		cvPutText(hw, "nacisnij spacje i czekaj trzymajac reke nieruchomo.", cvPoint( 10, 175 ), &font, color);
		cvShowImage("Proces kalibracji", hw);

		calibrateGest(pressedKey, capture, skinDetector, handFinder, faceDetector, functioneEaluation, 0, hw, font, color);

		cvReleaseImage(&preview);
		if(pressedKey == 27){ 
			cvDestroyWindow("Proces kalibracji");
			cvDestroyWindow("Podglad gestu");
			skinDetector.setActualFunction('0' + getBestFunction(functioneEaluation));
			return;
		}

		preview = cvLoadImage("a.jpg");
		if(preview == NULL){
			fprintf( stderr, "WARNING: No file a.jpg\n" );
		}
		cvShowImage("Podglad gestu", preview);

		cvSet(hw,cvScalar(0,0,0));
		cvPutText(hw, "Proces kalibracji", cvPoint( 10,30 ), &font, color);
		cvPutText(hw, "Etap trzeci - gest A:", cvPoint( 10, 70 ), &font, color);
		cvPutText(hw, "Pokaz przed kamera gest A (zacisnieta dlon) obok", cvPoint( 10, 100 ), &font, color);
		cvPutText(hw, "twarzy jak poprzednio, a nastepnie nacisnij spacje ", cvPoint( 10, 125 ), &font, color);
		cvPutText(hw, "i czekaj trzymajac reke nieruchomo.", cvPoint( 10, 150 ), &font, color);
		cvShowImage("Proces kalibracji", hw);

		calibrateGest(pressedKey, capture, skinDetector, handFinder, faceDetector, functioneEaluation, 1, hw, font, color);

		cvReleaseImage(&preview);
		if(pressedKey == 27){ 
			cvDestroyWindow("Proces kalibracji");
			cvDestroyWindow("Podglad gestu");
			skinDetector.setActualFunction('0' + getBestFunction(functioneEaluation));
			return;
		}

		preview = cvLoadImage("r.jpg");
		if(preview == NULL){
			fprintf( stderr, "WARNING: No file r.jpg\n" );
		}
		cvShowImage("Podglad gestu", preview);

		cvSet(hw,cvScalar(0,0,0));
		cvPutText(hw, "Proces kalibracji", cvPoint( 10,30 ), &font, color);
		cvPutText(hw, "Etap czwarty - gest R:", cvPoint( 10, 70 ), &font, color);
		cvPutText(hw, "Pokaz przed kamera gest R (wskazujacy i srodkowy palec ", cvPoint( 10, 100 ), &font, color);
		cvPutText(hw, "w gorze) obok twarzy jak poprzednio, a nastepnie nacisnij", cvPoint( 10, 125 ), &font, color);
		cvPutText(hw, "spacje i czekaj trzymajac reke nieruchomo.", cvPoint( 10, 150 ), &font, color);
		cvShowImage("Proces kalibracji", hw);

		calibrateGest(pressedKey, capture, skinDetector, handFinder, faceDetector, functioneEaluation, 2, hw, font, color);

		cvReleaseImage(&preview);
		if(pressedKey == 27){ 
			cvDestroyWindow("Proces kalibracji");
			cvDestroyWindow("Podglad gestu");
			skinDetector.setActualFunction('0' + getBestFunction(functioneEaluation));
			return;
		}

		preview = cvLoadImage("t.jpg");
		if(preview == NULL){
			fprintf( stderr, "WARNING: No file t.jpg\n" );
		}
		cvShowImage("Podglad gestu", preview);

		cvSet(hw,cvScalar(0,0,0));
		cvPutText(hw, "Proces kalibracji", cvPoint( 10,30 ), &font, color);
		cvPutText(hw, "Etap piaty  (ostatni) - gest T:", cvPoint( 10, 70 ), &font, color);
		cvPutText(hw, "Pokaz przed kamera gest T (kciuk i palec wskazujacy", cvPoint( 10, 100 ), &font, color);
		cvPutText(hw, "w ksztalcie litery T) obok twarzy jak poprzednio", cvPoint( 10, 125 ), &font, color);
		cvPutText(hw, "a nastepnie nacisnij spacje i czekaj trzymajac reke", cvPoint( 10, 150 ), &font, color);
		cvPutText(hw, "nieruchomo.", cvPoint( 10, 175 ), &font, color);

		cvShowImage("Proces kalibracji", hw);

		calibrateGest(pressedKey, capture, skinDetector, handFinder, faceDetector, functioneEaluation, 3, hw, font, color);
		skinDetector.setActualFunction('0' + getBestFunction(functioneEaluation));

		cvDestroyWindow("Podglad gestu");
		cvReleaseImage(&preview);

		cvSet(hw,cvScalar(0,0,0));
		cvPutText(hw, "Zakonczono", cvPoint( 10,30 ), &font, color);
		cvPutText(hw, "Aby rozpoczac dzialanie aplikacji nacisnij dowolny przycisk...", cvPoint( 10, 70 ), &font, color);
		cvPutText(hw, "UWAGA: Przydatne skroty w trakcie dzialania programu:", cvPoint( 10, 125 ), &font, color);
		cvPutText(hw, "spacja - rozpoczecie/zakonczenie przelozenia wynikow na", cvPoint( 10, 150 ), &font, color);
		cvPutText(hw, "         ruch i gesty myszka", cvPoint( 10, 175 ), &font, color);
		cvPutText(hw, " +/-  - zmiana czulosci ruchu dloni", cvPoint( 10, 200 ), &font, color);
		cvPutText(hw, " c     - ponowne pobranie statycznego tla", cvPoint( 10, 225 ), &font, color);
		cvPutText(hw, " 0-9   - reczna zmiana algorytmu detekcji skory", cvPoint( 10, 250 ), &font, color);
		cvPutText(hw, " s     - rozpoczecie/zakonczenie zapisywania obrazow po", cvPoint( 10, 275 ), &font, color);
		cvPutText(hw, "         segmentacji", cvPoint( 10, 300 ), &font, color);
		cvShowImage("Proces kalibracji", hw);

		waitKey(0);

		cvDestroyWindow("Proces kalibracji");
	}




}