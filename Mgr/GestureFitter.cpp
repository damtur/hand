#include "stdafx.h"
#include "GestureFitter.h"

namespace HandGR{
	static const int UNKNOWN = 0;
	static const int A = 1;
	static const int R = 2;
	static const int T = 3;

	GestureFitter::GestureFitter( int ACTIV_MOMENTS_COUNT, PositionDetector& positionDetector ) : ACTIV_MOMENTS_COUNT(ACTIV_MOMENTS_COUNT), positionDetector(positionDetector){
		ACTIV_MOMENTS_COUNT = 4;
		trainClassifiers();

		leftCounter = 0;
		middleCounter = 0;
		rightCounter = 0;

		mouseOn = false;
	}

	GestureFitter::~GestureFitter(){
		for(int i = 0; i < 4; ++i){
			delete meansData[i];
		}
	}

	void GestureFitter::trainClassifiers(){
		CvMat*  tempLoading = (CvMat*)cvLoad("trOpen.txt");
		if(tempLoading == NULL){
			fprintf( stderr, "Blad: Brak pliku trOpen.txt z zestawem uczacym...\n" );
			getchar();
			exit(EXIT_FAILURE);
		}
		Mat trOpen = Mat(tempLoading);


		tempLoading = (CvMat*)cvLoad("tra.txt");
		if(tempLoading == NULL){
			fprintf( stderr, "Blad: Brak pliku tra.txt z zestawem uczacym...\n" );
			getchar();
			exit(EXIT_FAILURE);
		}
		Mat trClose = Mat(tempLoading);

		tempLoading = (CvMat*)cvLoad("trr.txt");
		if(tempLoading == NULL){
			fprintf( stderr, "Blad: Brak pliku trr.txt z zestawem uczacym...\n" );
			getchar();
			exit(EXIT_FAILURE);
		}
		Mat trR = Mat(tempLoading);

		tempLoading = (CvMat*)cvLoad("trt.txt");
		if(tempLoading == NULL){
			fprintf( stderr, "Blad: Brak pliku trt.txt z zestawem uczacym...\n" );
			getchar();
			exit(EXIT_FAILURE);
		}
		Mat trOk = Mat(tempLoading);

		cv::Mat trSet = cv::Mat(trOpen.rows + trClose.rows + trR.rows + trOk.rows, ACTIV_MOMENTS_COUNT, CV_32FC1);
		trainingClassesResponses = cv::Mat(trOpen.rows + trClose.rows + trR.rows + trOk.rows, 1, CV_32SC1);

		cv::Mat meanMultiplier = cv::Mat(1,7,CV_64FC1);
		meanMultiplier.at<double>(0,0) = 1;
		meanMultiplier.at<double>(0,1) = 10;
		meanMultiplier.at<double>(0,2) = 100;
		meanMultiplier.at<double>(0,3) = 1000;
		meanMultiplier.at<double>(0,4) = 1000000;
		meanMultiplier.at<double>(0,5) = 10000;
		meanMultiplier.at<double>(0,6) = 1000000;

		int k = 0;

		//OPEN
		k = trainOneGest(trOpen, k, 0, trSet, trainingClassesResponses, meanMultiplier);
		//CLOSE
		k = trainOneGest(trClose, k, 1, trSet, trainingClassesResponses, meanMultiplier);
		//R
		k = trainOneGest(trR, k, 2, trSet, trainingClassesResponses, meanMultiplier);
		//OK
		k = trainOneGest(trOk, k, 3, trSet, trainingClassesResponses, meanMultiplier);

#ifdef GESTURE_FITTER_GPU
#ifdef GPUON
		bool gpuEnable = gpu::getCudaEnabledDeviceCount() != 0;
#else	
		bool gpuEnable = false;
#endif
#else
		bool gpuEnable = false;
#endif
		if(gpuEnable){
			trainGpuMacher(trSet);
		}else{
			bayesClassifier.train(trSet, trainingClassesResponses);
			nerestClassifier.train(trSet, trainingClassesResponses);

			//svmClassiefierOC.train(trSet.rowRange(0,trOpen.rows+trClose.rows), responses.rowRange(0,trOpen.rows+trClose.rows));
			svmClassiefierAR.train(trSet.rowRange(trOpen.rows,trOpen.rows+trClose.rows+trR.rows), trainingClassesResponses.rowRange(trOpen.rows,trOpen.rows+trClose.rows+trR.rows));
			svmClassiefierRK.train(trSet.rowRange(trOpen.rows+trClose.rows,trOpen.rows+trClose.rows+trR.rows+trOk.rows), trainingClassesResponses.rowRange(trOpen.rows+trClose.rows,trOpen.rows+trClose.rows+trR.rows+trOk.rows));
			k = 0;

			k = trainOneGest(trR, k, 2, trSet, trainingClassesResponses, meanMultiplier);
			//CLOSE
			k = trainOneGest(trOpen, k, 0, trSet, trainingClassesResponses, meanMultiplier);
			//R
			k = trainOneGest(trOk, k, 3, trSet, trainingClassesResponses, meanMultiplier);
			////OK
			k = trainOneGest(trClose, k, 1, trSet, trainingClassesResponses, meanMultiplier);


			svmClassiefierRO.train(trSet.rowRange(0,trR.rows+trOpen.rows), trainingClassesResponses.rowRange(0,trR.rows+trOpen.rows));
			//svmClassiefierOK.train(trSet.rowRange(trR.rows,trR.rows+trOpen.rows+trOk.rows), responses.rowRange(trR.rows,trR.rows+trOpen.rows+trOk.rows));
			//svmClassiefierKC.train(trSet.rowRange(trR.rows+trOpen.rows,trR.rows+trOpen.rows+trOk.rows+trClose.rows), responses.rowRange(trR.rows+trOpen.rows,trR.rows+trOpen.rows+trOk.rows+trClose.rows));
		}
	}

	int GestureFitter::trainOneGest(const Mat &trSetSmall, int sampleCount, const int k, cv::Mat &trSet, cv::Mat &responses, const cv::Mat &meanMultiplier ){
		meansData[k] = new double[ACTIV_MOMENTS_COUNT];
		for(int i = 0; i < ACTIV_MOMENTS_COUNT; ++i){ meansData[k][i] = 0;}
		for(int i = 0 ; i < trSetSmall.rows; ++ i, ++sampleCount){
			for(int j = 0 ; j < ACTIV_MOMENTS_COUNT; ++ j){
				trSet.at<float>(sampleCount,j) = (float)trSetSmall.at<double>(i,j);
				meansData[k][j] += trSetSmall.at<double>(i,j) * meanMultiplier.at<double>(0, j);
			}
			responses.at<int>(sampleCount) = k;
		}
		for(int i = 0; i < ACTIV_MOMENTS_COUNT; ++i){ meansData[k][i] /= trSetSmall.rows; }
		huMomentsMean[k] = cv::Mat(1,ACTIV_MOMENTS_COUNT, CV_64FC1, meansData[k]);
		return sampleCount;
	}

	void GestureFitter::trainGpuMacher( cv::Mat trSet ){
		vector<gpu::GpuMat> gpuTrainDescriptorCollection;
		Mat tempDescriptor = Mat(1, ACTIV_MOMENTS_COUNT, CV_32FC1);
		//Mat tempDescriptor = Mat(1, 3, CV_32FC1);

		for(int i = 0; i < trSet.rows; ++i){
			for(int j = 0 ; j < ACTIV_MOMENTS_COUNT; ++ j){
				tempDescriptor.at<float>(j) = trSet.at<float>(i, j);
			}
			gpu::GpuMat tempGpuDescriptor(tempDescriptor);
			gpuTrainDescriptorCollection.push_back(tempGpuDescriptor);
		}
		gpuMacher.add(gpuTrainDescriptorCollection);
	}

	bool GestureFitter::mapResultToMouse(int nearestClassifierResult, int knnClassifierresult, int toFind){
		if(nearestClassifierResult == knnClassifierresult){
			switch(nearestClassifierResult){
			case UNKNOWN:
				return handleUnknown(toFind);
			case A:
				return handleA(toFind);
			case R:
				return handleR(toFind);
			case T:
				return handleT(toFind);
			}
		}
		return false;
	}
	
	bool GestureFitter::mapResultToMouse( int bayesClassifierResult, int nearest1ClassifierResult, int nearest3ClassifierResult, int svmClassifierARResult, int svmClassifierRKResult, int toFind ){
		if(bayesClassifierResult == nearest1ClassifierResult && nearest1ClassifierResult == nearest3ClassifierResult){
			if(bayesClassifierResult == UNKNOWN){
				return handleUnknown(toFind);
			}else if ( bayesClassifierResult == A){
				if(svmClassifierARResult == A){
					return handleA(toFind);
				}else{
					if(toFind != -1) return false;
				}
			}else if (bayesClassifierResult == R){
				if(svmClassifierARResult == R && /*ro == 2 && */svmClassifierRKResult == R){
					return handleR(toFind);
				}else{
					if(toFind != -1) return false;	
				}
			}else {//bay ==3
				if(svmClassifierRKResult == T){
					return handleT(toFind);

				}else{
					if(toFind != -1) return false;
					//cout<<"N";	
				}
			}
		}else{
			if(toFind != -1) return false;
			//cout<<"N";	
			//leftCounter = 0;
			//middleCounter = 0;
			//rightCounter = 0;				
		}
		return false;
	}


	//////////////////////////////////////////////////////////////////////////
	/// PUBLIC
	//////////////////////////////////////////////////////////////////////////

	bool GestureFitter::fitGesture( Mat& imageWithMarks, const Moments &winerMom, const double winnerHuMoments[7], int toFind, int winnerConturId, bool isCalibration, const vector<vector<Point> >& contours ){
		int size = static_cast<int>(sqrt(winerMom.m00)/2);
		int xMean = static_cast<int>(winerMom.m10/winerMom.m00);
		int yMean = static_cast<int>(winerMom.m01/winerMom.m00);
		if(xMean - size > 10 && xMean + size < RESOLUTION_X - 10 && yMean - size > 10 && yMean + size < RESOLUTION_Y - 10 ){
			bool resultedGest = findNearestGesture(winnerHuMoments, toFind);

			if(toFind != -1){
				return resultedGest;
			}

			positionDetector.renewPositionPoints(xMean, yMean, size, isCalibration);

			//Utils::globalToSave.at<float>(Utils::globalCounter) = (((double)getTickCount() - Utils::globalTime)/getTickFrequency());
			//++Utils::globalCounter;
		}else{
			if(toFind != -1)
				return false;
		}

#ifdef DRAW_HAND_RESULT
		if(!isCalibration){
			drawContours(imageWithMarks, contours, winnerConturId, Scalar(0,255,0), 1);
			rectangle(imageWithMarks, Point(xMean-size, yMean-size), Point(xMean+size, yMean+size), Scalar(255,128,0,0));
		}
#endif

		return false;
	}


	bool GestureFitter::findNearestGesture( const double * winnerHuMoments, int toFind ){
		cv::Mat clasesMat = cv::Mat(1,ACTIV_MOMENTS_COUNT, CV_32FC1);
		for(int i = 0; i < ACTIV_MOMENTS_COUNT; ++i){
			clasesMat.at<float>(0,i) = (float)(winnerHuMoments[i]);
		}
		int bayesClassifierResult = static_cast<int>(bayesClassifier.predict(clasesMat)) ;
		int nearest1ClassifierResult = static_cast<int>(nerestClassifier.find_nearest(clasesMat,1));
		int nearest3ClassifierResult = static_cast<int>(nerestClassifier.find_nearest(clasesMat,3));
		//int bos = static_cast<int>(svmClassiefierRK.predict(clasesMat));

		//int oc = static_cast<int>(svmClassiefierOC.predict(clasesMat));
		//int ok = static_cast<int>(svmClassiefierOK.predict(clasesMat));
		//int ro = static_cast<int>(svmClassiefierRO.predict(clasesMat));
		//int kc = static_cast<int>(svmClassiefierKC.predict(clasesMat));
		int svmClassifierARResult = static_cast<int>(svmClassiefierAR.predict(clasesMat));
		int svmClassifierRKResult = static_cast<int>(svmClassiefierRK.predict(clasesMat));
		//int tre = static_cast<int>(decisionTreeClassifier.predict(clasesMat)->value);

#ifdef PRINT_RESULT
		printResult(bayesClassifierResult, nearest1ClassifierResult, nearest3ClassifierResult, svmClassifierARResult, svmClassifierRKResult);
#endif
		return mapResultToMouse(bayesClassifierResult, nearest1ClassifierResult, nearest3ClassifierResult, svmClassifierARResult, svmClassifierRKResult, toFind);
	}



	//////////////////////////////////////////////////////////////////////////
	/// GPU
	//////////////////////////////////////////////////////////////////////////


	bool GestureFitter::gpuFitGesture(const gpu::GpuMat& grayFrame, Mat& imageWithMarks, const Moments &winerMom, const double winnerHuMoments[7], int toFind, int winnerConturId, bool isCalibration, const vector<vector<Point> >& contours ){

		int size = static_cast<int>(sqrt(winerMom.m00)/2);
		int xMean = static_cast<int>(winerMom.m10/winerMom.m00);
		int yMean = static_cast<int>(winerMom.m01/winerMom.m00);
		if(xMean - size > 10 && xMean + size < RESOLUTION_X - 10 && yMean - size > 10 && yMean + size < RESOLUTION_Y - 10 ){
#ifdef GESTURE_FITTER_GPU
			bool resultedGest = gpuFindNearestGesture(winnerHuMoments, toFind);
#else
			bool resultedGest = findNearestGesture(winnerHuMoments, toFind);
#endif


#ifdef DRAW_HAND_RESULT
			if(!isCalibration){
				drawContours(imageWithMarks, contours, winnerConturId, Scalar(0,255,0), 1);
				rectangle(imageWithMarks, Point(xMean-size, yMean-size), Point(xMean+size, yMean+size), Scalar(255,128,0,0));
			}
#endif

			if(toFind != -1){
				return resultedGest;
			}

			positionDetector.renewPositionPoints(grayFrame, xMean, yMean, size);

			//Utils::globalToSave.at<float>(Utils::globalCounter) = (((double)getTickCount() - Utils::globalTime)/getTickFrequency());
			//++Utils::globalCounter;
		}else{
			if(toFind != -1)
				return false;
			//cout<<"N";	
		}

		return false;
	}



	bool GestureFitter::gpuFindNearestGesture( const double * winnerHuMoments, int toFind ){
		cv::Mat clasesMat = cv::Mat(1,ACTIV_MOMENTS_COUNT, CV_32FC1);
		for(int i = 0; i < ACTIV_MOMENTS_COUNT; ++i){
			clasesMat.at<float>(0,i) = (float)(winnerHuMoments[i]);
		}

		gpu::GpuMat query(clasesMat);

		vector<DMatch> gpuMacherResultNearest;
		vector<vector<DMatch> > gpuMacherResultKnn;
		gpuMacher.match(query, gpuMacherResultNearest);

		int nearestMacherResult = gpuMacherResultNearest.empty() ? UNKNOWN : trainingClassesResponses.at<int>(gpuMacherResultNearest[0].imgIdx);
		int knnMacherResult = nearestMacherResult;

#ifdef PRINT_RESULT
		cout<<"Macher: "<<nearestMacherResult<<"  "<<endl;
#endif

		return mapResultToMouse(nearestMacherResult, knnMacherResult, toFind);
	}


	void GestureFitter::printResult( int bayesClassifierResult, int nearest1ClassifierResult, int nearest3ClassifierResult, int svmClassifierARResult, int svmClassifierRKResult ){
		cout<<getStringFor(bayesClassifierResult)<<
			getStringFor(nearest1ClassifierResult)<<
			getStringFor(nearest3ClassifierResult)<<
			getStringFor(svmClassifierARResult)<<
			getStringFor(svmClassifierRKResult)<<
			endl;
	}

	cv::string GestureFitter::getStringFor( int i ){
		switch(i){
		case UNKNOWN:
			return "?";
		case A:
			return "A";
		case R:
			return "R";
		case T: 
			return "T";
		default:
			return "Nothing";
		}
	}

	bool GestureFitter::handleA( int toFind ){
		if(toFind != -1) return (toFind == 1);
		//cout << "A";
		if(++middleCounter >= 3){
			//cout << "ROZPOZNANO!: A"<<endl;
			if(mouseOn){
				OutputServer::getInstance()->gestA();
			}
			middleCounter = 0;
		}
		leftCounter = rightCounter = 0;
		return false;
	}

	bool GestureFitter::handleR( int toFind ){
		if(toFind != -1) return (toFind == 2);
		//cout << "R";
		if(++leftCounter >= 3){
			//cout << "ROZPOZNANO!: R"<<endl;
			if(mouseOn){
				OutputServer::getInstance()->gestR();
			}
			leftCounter = 0;
		}
		middleCounter = rightCounter = 0;
		return false;
	}

	bool GestureFitter::handleUnknown( int toFind ){
		//if(ro == 0){
		if(toFind != -1) return (toFind == 0);
		//cout<<"W";
		if(mouseOn){
			OutputServer::getInstance()->gestNothing();
		}
		leftCounter = middleCounter = rightCounter = 0;
		//}
		return false;
	}

	bool GestureFitter::handleT( int toFind ){
		if(toFind != -1) return (toFind == 3);
		///cout << "T";
		if(++rightCounter >= 3){
			//cout << "ROZPOZNANO!: T"<<endl;
			if(mouseOn){
				OutputServer::getInstance()->gestT();
			}
			rightCounter = 0;
		}
		leftCounter = middleCounter = 0;
		return false;
	}

	

}