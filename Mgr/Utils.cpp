#include "stdafx.h"

namespace HandGR{
	int Utils::globalCounter = 0;
	Mat Utils::globalToSave = Mat(MAX_TEST_ITERATIONS,1, CV_32FC1);
	int Utils::globalFunction = '0';
	double Utils::globalTime = 0.0;

}