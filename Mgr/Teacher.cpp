#include "stdafx.h"

namespace HandGR{

	Teacher::Teacher(unsigned maxSamples, unsigned attributeCount/* = 7*/, float * multiplier/* = NULL*/){

		this->attributeCount = attributeCount;
		this->maxSamples = maxSamples;

		currentSampleNr = 0;
		trainArray = cvCreateMat(maxSamples, attributeCount, CV_32FC1);

		if(multiplier == NULL){
			this->multiplier = new float[attributeCount];
			for(unsigned i = 0; i < attributeCount ; ++i){
				this->multiplier[i] = 1;
				// 					mean[1]*=10;
				// 					mean[2]*=100;
				// 					mean[3]*=1000;
				// 					mean[4]*=1000000;
				// 					mean[5]*=10000;
				// 					mean[6]*=1000000;
			}
		}else{
			this->multiplier = multiplier;
		}
	}

	Teacher::~Teacher(){
		cvReleaseMat(&trainArray);
		delete multiplier;
	}

	void Teacher::setMultiplier(float * multiplier){
		if(this->multiplier != NULL)
			delete multiplier;
		this->multiplier = multiplier;
	}

	void Teacher::saveMean(const char * filename, float* multiplier/* = NULL*/){
		if(multiplier == NULL)
			multiplier = this->multiplier;
		float* mean = new float[attributeCount];

		for(unsigned i = 0; i < currentSampleNr ; ++i){
			for(unsigned j = 0; j < attributeCount; ++j){
				mean[j] += trainArray->data.fl[i*7 + j];;
			}	
		}

		for(unsigned i = 0; i < attributeCount; ++i){
			mean[i] /= currentSampleNr;
			mean[i] *= multiplier[i];
		}


		CvMat tempMat = cvMat(1,7,CV_32FC1,mean);
		cvSave(filename, &tempMat);

		delete mean;
	}



	unsigned Teacher::train(float* vct, const char * filename/* = NULL*/){
		for(unsigned i = 0; i < attributeCount; ++i){
			trainArray->data.fl[currentSampleNr*attributeCount + i] = vct[i];
		}
		if(currentSampleNr + 1 < maxSamples)
			++currentSampleNr;
		else
			if(filename != NULL){
				saveTrain(filename);
				currentSampleNr = 0;
			}
			return currentSampleNr+1;
	}


	double Teacher::calcDistance(const double* huMoments, const cv::Mat& mean){

		cv::Mat covar=  cv::Mat(7,7,CV_64FC1);
		cv::Mat mea = cv::Mat(7,7,CV_64FC1);

		cv::Mat inputMatrix[2];

		double hus[] = {huMoments[0], huMoments[1]*10, huMoments[2]*100, huMoments[3]*1000, huMoments[4]*1000000, huMoments[5]*10000, huMoments[6]*1000000 };
		cv::Mat clasesMat = cv::Mat(1,7, CV_64FC1, hus);

		inputMatrix[0] = mean;
		inputMatrix[1] = clasesMat;

		cv::calcCovarMatrix(inputMatrix, 2, covar, mea, CV_COVAR_NORMAL);

		cv::Mat covarInv = cv::Mat(7,7, CV_64FC1);
		cv::invert(covar, covarInv, cv::DECOMP_SVD);


		//ERROR IN OPEN CV!! always return 1.41421 - return cv::Mahalanobis(mean, clasesMat, covarInv);
		return Utils::myMahalanobis(mean, clasesMat, covarInv);
	}

	// Funkcja s³u¿y do obliczania cech z obrazow i zapisywania ich w pliku
	void Teacher::processImages(const string sourceFolder, const unsigned int count, const string destFilename, const unsigned momentsCount/*=7*/, const string extenstion/*="bmp"*/){
		IplImage* dst = NULL;
		vector<vector<cv::Point> > contours;
		double huMoments[7];

		//double* meanData = new double[7];

		//meanData[0] = 0.22796999;
		//meanData[1] = 0.10783564;
		//meanData[2] = 0.21251296;
		//meanData[3] = 0.25522405;
		//meanData[4] = 0.33939978;
		//meanData[5] = 0.21461283;
		//meanData[6] = 0.14835256;
		//cv::Mat mean = cv::Mat(1,7, CV_64FC1, meanData);

		Ptr<CvMemStorage> storage = cvCreateMemStorage(0);

		unsigned savedCounter = 0;
		cv::Mat momentsToSave = cv::Mat(count, momentsCount, CV_64FC1);

		Moments moments;

		for(unsigned i = 0; i < count; ++ i){
			stringstream str; 
			str<< sourceFolder << "/tr(" << i << ")."<<extenstion;

			IplImage *temp = cvLoadImage(str.str().c_str());
			Mat image = cv::Mat(temp);
			//Mat image = cv::imread(str.str());


			if(dst == NULL)
				dst = cvCreateImage( image.size(), 8, 3 );

			cvZero( dst );

			vector<Mat> splites;
			split(image, splites);

			findContours(splites[0], contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
			HandFinder::correctHandArea(contours);


			for(unsigned conturId = 0 ; conturId < contours.size(); ++conturId){
				Moments tempMoments = cv::moments(contours[conturId], true);

				if ( tempMoments.m00 > 1000){
					cv::HuMoments(tempMoments, huMoments);
					//double tempDistance = calcDistance(huMoments, mean);
					//if(abs(tempDistance) < 0.09){

					cv::Mat clasesMat = cv::Mat(1,1, CV_32FC1, huMoments);

					for(unsigned j = 0 ; j < momentsCount; ++j){
						momentsToSave.at<double>(savedCounter, j) = huMoments[j];
					}
					++savedCounter;
					if(savedCounter >= count)
						break;
					//}

				}
			}//for contures

			if(savedCounter >= count)
				break;	
		}//for all images


		CvMat tempMat = (CvMat)momentsToSave;

		cvSave(destFilename.c_str(), &tempMat );

		//delete meanData;
	}


	void Teacher::saveTrain(const char * filename){
		cvSave(filename, trainArray);
	}

}