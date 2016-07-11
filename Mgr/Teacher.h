#pragma once
#include "stdafx.h"

using namespace cv;
using namespace std;

namespace HandGR{
	class Teacher {

		unsigned attributeCount;
		unsigned maxSamples;
	
		unsigned currentSampleNr;
		float * multiplier;

		CvMat* trainArray;
	public:
		
		/* Init teacher with max samples and attribute count parameters */
		Teacher(unsigned maxSamples, unsigned attributeCount = 7, float * multiplier = NULL);

		~Teacher();

		/** Set multipler for feature */
		void setMultiplier(float * multiplier);
		
		/** Save mean value of all train set */
		void saveMean(const char * filename, float* multiplier = NULL);

		unsigned train(float* vct, const char * filename = NULL);

		/** Calc mahalanobis distance */
		static double calcDistance(const double* huMoments, const cv::Mat& mean);

		// Funkcja s³u¿y do obliczania cech z obrazow i zapisywania ich w pliku
		static void processImages(const string sourceFolder, const unsigned int count, const string destFilename, const unsigned momentsCount=7, const string extenstion="bmp");

		private:
			void saveTrain(const char * filename);
	};
}