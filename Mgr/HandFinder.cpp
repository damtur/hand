#include "stdafx.h"
#include "HandFinder.h"

namespace HandGR{

	int zonk = 0;
	bool on= false;
	HandFinder::HandFinder(): gestureFitter(7, positionDetector) /*, teacher(500, 7)*/{

		//positionDetector = PositionDetector();TODO CZY TO DZIA£A??

		saving = false;
		save = false;
		stop = false;
		//counter = 0;
		type = 0;

		//mouse moving support
		srand ( (unsigned)time(NULL) );

		//Faster hand finder (ROI)
		prevRoiX = 0;
		prevRoiY = 0;

		lastFrameTime = clock();
		frameDuration = 1000;

	}


	//////////////////////////////////////////////////////////////////////////
	// UTILS
	//////////////////////////////////////////////////////////////////////////
	double HandFinder::calcDistance(double* huMoments, const Mat& mean){

		cv::Mat covar=  cv::Mat(gestureFitter.getActiveMomentsCount(),gestureFitter.getActiveMomentsCount(),CV_64FC1);
		cv::Mat mea = cv::Mat(gestureFitter.getActiveMomentsCount(),gestureFitter.getActiveMomentsCount(),CV_64FC1);

		cv::Mat inputMatrix[2];

		double hus[] = {huMoments[0], huMoments[1]*10, huMoments[2]*100, huMoments[3]*1000, huMoments[4]*1000000, huMoments[5]*10000, huMoments[6]*1000000 };
		cv::Mat clasesMat = cv::Mat(1,gestureFitter.getActiveMomentsCount(), CV_64FC1, hus);

		inputMatrix[0] = mean;
		inputMatrix[1] = clasesMat;

		cv::calcCovarMatrix(inputMatrix, 2, covar, mea, CV_COVAR_NORMAL);

		cv::Mat covarInv = cv::Mat(gestureFitter.getActiveMomentsCount(),gestureFitter.getActiveMomentsCount(), CV_64FC1);
		cv::invert(covar, covarInv, cv::DECOMP_SVD);


		//ERROR IN OPEN CV!! always return 1.41421 that must write own function
		//return cv::Mahalanobis(mean, clasesMat, covarInv);
		return Utils::myMahalanobis(mean, clasesMat, covarInv);
	}

	void HandFinder::processKey( const int pressedKey ){
		if(pressedKey == 'a'){
			saving=true;
			type = 0;
		}else if(pressedKey == 's'){
			saving = true;
			type = 1;
		}else if(pressedKey == 'd'){
			saving = true;
			type = 2;
		}else if(pressedKey == 'm'){
			stop = true;
		}else if(pressedKey == '+' || pressedKey == '=' ){
			OutputServer::getInstance()->increaseSensitivity();
		}else if(pressedKey == '-' || pressedKey == '_'){
			OutputServer::getInstance()->decreaseSensitivity();
		}else if(pressedKey == ' '){
			gestureFitter.setMouseOn(gestureFitter.isMouseOn() ? false : true);
		}
	}


	void HandFinder::correctHandArea(vector<vector<Point> >& contours ){
		//find extremums
		for(vector<vector<Point> >::iterator contourIt = contours.begin(); contourIt!=contours.end(); ++contourIt){

			////Line fitting
			//Vec4f temp;
			//fitLine(*contourIt, temp, CV_DIST_L2, 0, 0.01, 0.01);
			//Point second = Point(temp[0]*10 + temp[2], temp[1]*10+temp[3]);
			//line(imageWithMarks, Point(temp[2], temp[3]), second, Scalar(255,255,255), 2);

			//fitEllipse
			if(contourIt->size() <= 5){
				//contourIt = contours.erase(contourIt);
				//if(contourIt == contours.end()){
				//	break;
				//}
				continue;
			}

			RotatedRect r = fitEllipse(*contourIt);

			//Get rid of to small objects!
			if(r.size.width * r.size.height < 1000){
				//contourIt = contours.erase(contourIt);
				//if(contourIt == contours.end()){
				//	break;
				//}
				continue;
			}
			//cout<<r.angle-180<<endl;
			//ellipse(imageWithMarks, r, Scalar(255,255,255),2, 4);
			
			Mat m = getRotationMatrix2D(Point(0,0), r.angle - 180, 1);
			Mat minv = getRotationMatrix2D(Point(0,0), -r.angle + 180, 1);
			transform(*contourIt, *contourIt, m);

			Rect rect = boundingRect(*contourIt);
			int maxHeight = (int)(rect.width * 1.2);
			int maxY = rect.tl().y + maxHeight;

			//int previousX = 0;
			Point previousPt = *(contourIt->rbegin());
			if(maxHeight < rect.height){
				for(vector<Point>::iterator pointIt = contourIt->begin(); pointIt != contourIt->end(); ){
					if(pointIt->y > maxY){
						if(previousPt.y > maxY){
							pointIt = contourIt->erase(pointIt);
							if(pointIt == contourIt->end()){
								break;
							}
							previousPt = *pointIt;
							continue;
						}else{
							//previous good
							//cout<<"MAX: " << maxY << " prevY" << previousPt.y << " currY" << pointIt->y << endl;
							pointIt->x = previousPt.x + (maxY - previousPt.y)*(pointIt->x - previousPt.x)/(pointIt->y - previousPt.y);
							pointIt->y = maxY;
						}
					}else{
						if(previousPt.y > maxY){
							pointIt->x += (maxY - pointIt->y)*(previousPt.x - pointIt->x)/(previousPt.y - pointIt->y );
							pointIt->y = maxY;
						}//else this is good
					}
					previousPt = *pointIt;
					++pointIt;
				}

				//	if(pointIt->y > maxY){
				//		if(previousX < pointIt->x){
				//			pointIt->y = maxY;
				//			previousX = pointIt->x;
				//		}else{
				//			pointIt = contourIt->erase(pointIt);
				//			if(pointIt == contourIt->end()){
				//				break;
				//			}
				//			continue;
				//		}
				//	}
				//	++ pointIt;
				//}
			}
			transform(*contourIt, *contourIt, minv);
		}
	}


	void HandFinder::drawHandArea( vector<vector<Point> > &contours, Mat& imageWithMarks ){
		for(unsigned conturId = 0 ; conturId < contours.size(); ++conturId){
			for(unsigned ptsId = 0; ptsId < contours[conturId].size(); ++ptsId){
				circle(imageWithMarks, contours[conturId][ptsId], 1, Scalar(255, 0, 255));
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////
	// HAND FINDER
	//////////////////////////////////////////////////////////////////////////
	bool HandFinder::findHand( Mat& imageAfterSkinRecognition, const Mat& frameImage, Mat& imageWithMarks, const int pressedKey, FaceDetector& faceDetector, bool isCalibration, int toFind /*= -1*/ ){

		clock_t currentTime = clock();
		frameDuration = static_cast<long>(currentTime - lastFrameTime);
		lastFrameTime = clock();

		processKey(pressedKey);

		//Morphology
		cv::erode(imageAfterSkinRecognition, imageAfterSkinRecognition, cv::Mat(), cv::Point(-1,-1), 1);//2);1
		cv::dilate(imageAfterSkinRecognition, imageAfterSkinRecognition, cv::Mat(), cv::Point(-1,-1), 9);//7);3
		cv::erode(imageAfterSkinRecognition, imageAfterSkinRecognition, cv::Mat(), cv::Point(-1,-1), 8);//5);2

		//Detect position of hand and move cursor 
		positionDetector.detectPosition(frameImage, isCalibration, gestureFitter.isMouseOn(), frameDuration);
#ifdef PRITNT_MOVE_POINTS
		positionDetector.printPoints(imageWithMarks);
#endif
		vector<Mat> splites;
		split(imageAfterSkinRecognition, splites);
		
		//CPU ONLY
		cv::findContours(splites[0], contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		correctHandArea(contours);

#ifdef DRAW_HAND_RESULT
		drawHandArea(contours, imageWithMarks);
#endif

		int winnerGest;
		Moments winerMoments;
		int winnerConturId;
		bool found = findBestHandArea(faceDetector, winnerGest, winnerConturId, winerMoments, huMoments,  contours);

		if(found){
			bool temp = gestureFitter.fitGesture(imageWithMarks, winerMoments, huMoments, toFind, winnerConturId, isCalibration, contours);
			return temp;
		}else{//not found
			if(toFind != -1)
				return false;
		}

		return found;
	}

	bool HandFinder::findBestHandArea( FaceDetector& faceDetector, int &winnerGest, int &winnerConturId, Moments& winerMoments, double winnerHuMoments[7], vector<vector<Point> >& contours ){
		bool found = false;
		for(unsigned conturId = 0 ; conturId < contours.size(); ++conturId){
			double minDistance = DBL_MAX;
			
			Moments tempMoments = cv::moments(contours[conturId], true);
			double tempHuMoments[7];

			int positionXMean = static_cast<int>(tempMoments.m10/tempMoments.m00);
			int positionYMean = static_cast<int>(tempMoments.m01/tempMoments.m00);

			bool inFace = faceDetector.inFace(positionXMean, positionYMean);

			if ( !inFace && tempMoments.m00 > 500){
				cv::HuMoments(tempMoments, tempHuMoments);

				double tempDistance[4];
				tempDistance[0] = calcDistance(tempHuMoments, gestureFitter.getHuMomentsMean()[0]);
				tempDistance[1] = calcDistance(tempHuMoments, gestureFitter.getHuMomentsMean()[1]);
				tempDistance[2] = calcDistance(tempHuMoments, gestureFitter.getHuMomentsMean()[2]);
				tempDistance[3] = calcDistance(tempHuMoments, gestureFitter.getHuMomentsMean()[3]);

				double tempMinDistance = tempDistance[0];
				int tempWinnerGest = 0;
				for(int i = 1; i < 4; ++i){
					if(tempDistance[i] < tempMinDistance){
						tempWinnerGest = i;
						tempMinDistance = tempDistance[i];
					}
				}

				if(abs(tempMinDistance) < 0.051){
					if(found == false ){
						minDistance = abs(tempMinDistance);
						winnerGest = tempWinnerGest;
						//moments = tempMoments;
						for(int i = 0; i < 7; ++i)
							winnerHuMoments[i] = tempHuMoments[i];
						found = true;
						winnerConturId = conturId;
						winerMoments = tempMoments;
					}else{
						if(abs(tempMinDistance) < minDistance ){
							minDistance = tempMinDistance;
							winnerGest = tempWinnerGest;
							//moments = tempMoments;
							for(int i = 0; i < 7; ++i)
								winnerHuMoments[i] = tempHuMoments[i];
							winnerConturId = conturId;
							winerMoments = tempMoments;
						}
					}
				}
			}
		}
		return found;
	}

	
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU HAND FINDER
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	bool HandFinder::gpuFindHand( const gpu::GpuMat afterSkinDetection, const gpu::GpuMat grayFrame, vector<gpu::GpuMat>& chanels, gpu::GpuMat buf, Mat& imageWithMarks, const int pressedKey, FaceDetector& faceDetector, bool isCalibration, gpu::Stream& gpuStream, int toFind /*= -1*/ ){

		clock_t currentTime = clock();
		frameDuration = static_cast<long>(currentTime - lastFrameTime);
		lastFrameTime = clock();

		processKey(pressedKey);

		//Morphology to avoid distractions
		gpu::split(afterSkinDetection, chanels, gpuStream);

		Utils::binaryOpen(chanels[0], chanels[0], buf, 1, gpuStream, STRUCT_TYPE_RECT);
		Utils::binaryClose(chanels[0], chanels[0], buf, 8, gpuStream, STRUCT_TYPE_RECT);

		//Detect position of hand and move cursor 
		positionDetector.detectPosition(grayFrame, isCalibration, gestureFitter.isMouseOn(), frameDuration);
#ifdef PRITNT_MOVE_POINTS
		positionDetector.printPoints(imageWithMarks);
#endif
	
		//Download prepared image from GPU
		Mat afterSkinDetectionGray(chanels[0].rows, chanels[0].cols, chanels[0].type());
		gpuStream.enqueueDownload(chanels[0], afterSkinDetectionGray);


		//CPU ONLY
		cv::findContours(afterSkinDetectionGray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		correctHandArea(contours);

		int winnerGest;
		Moments winerMoments;
		int winnerConturId;
		bool found = findBestHandArea(faceDetector, winnerGest, winnerConturId, winerMoments, huMoments, contours);

		if(found){

			gestureFitter.gpuFitGesture(grayFrame, imageWithMarks, winerMoments, huMoments, toFind, winnerConturId, isCalibration, contours);

			return found;
		}else{//not found
			if(toFind != -1)
				return false;
			//cout<<"N";
		}
		

		return found;
	}

};
