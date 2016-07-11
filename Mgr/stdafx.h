#pragma once
#pragma warning( disable: 4996 )


#ifndef MAX
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef MAXI
#define MAXI(a,b,c)			(((a) > (b)) ? (((a) > (c)) ? (a) : (c)) : (((b) > (c)) ? (b) : (c)))
#endif

#ifndef MIN
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifndef MINI
#define MINI(a,b,c)			(((a) < (b)) ? (((a) < (c)) ? (a) : (c)) : (((b) < (c)) ? (b) : (c)))
#endif

#ifndef FIT_RANGE
#define FIT_RANGE(v,l,h) (MIN(h, MAX(l, v)))
#endif

#define POSITION_DRAW_POINTS
#define PRINT_RESULT
#define MAX_TEST_ITERATIONS 200
#define DRAW_HAND_RESULT
#define SHOW_IMAGES

//#define GESTURE_FITTER_GPU don't enable :)

#define GPUON

//Constants
const unsigned RESOLUTION_X = 1270;
const unsigned RESOLUTION_Y = 720;

const unsigned SERVER_PORT = 8879;

static const char DATAGRAM_MOVE = 0;
static const char DATAGRAM_R = 1;
static const char DATAGRAM_A = 2;
static const char DATAGRAM_T = 3;
static const char DATAGRAM_NOTHING = 4;
static const char DATAGRAM_INCREASE_SENSITIVITY = 5;
static const char DATAGRAM_DECREASE_SENSITIVITY = 6;

static const unsigned int TIMEOUT_COMMON = 20;


//Windows
#include <winsock2.h>
#include <windows.h>
#include "targetver.h"
 

//Stl
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <math.h>
#define _USE_MATH_DEFINES
#include <iomanip>
#include <sstream>
#include <time.h>
#include <limits>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>

//OpenCV
#include <opencv2/imgproc/imgproc.hpp>
//#include <cvaux.h>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/contrib/contrib.hpp>

//GPU
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#include <cuda.h>

#include "GpuSkinDetector.h"
#include "GpuFunctions.h"

//My functions
#include "OutputServer.h"
#include "GpuFrames.h"
#include "Utils.h"
//#include "MouseHandling.h"

#include "SkinDetector.h"
#include "Teacher.h"
#include "PositionDetector.h"
#include "FaceDetector.h"
#include "GestureFitter.h"
#include "HandFinder.h"
#include "Calibration.h"
#include "StereoCalibration.h"


