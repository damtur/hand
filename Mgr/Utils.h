#pragma once
#include "stdafx.h"

using namespace cv;
using namespace std;


#define ASD_RGB_SET_PIXEL(pointer, r, g, b)	{ (*pointer) = (unsigned char)b; (*(pointer+1)) = (unsigned char)g;	(*(pointer+2)) = (unsigned char)r; }
namespace HandGR{
	/** Some static functions used by the program */
	class Utils{
	public:
		static int globalCounter;
		static Mat globalToSave;
		static int globalFunction;
		static double globalTime;

		static void globalSaveToFile(String str = ""){
			
			if(str.length() == 0){
				ostringstream ss;

#ifdef GPUON
				ss << "GPU" << RESOLUTION_X << "x" <<RESOLUTION_Y << "Fun" << (Utils::globalFunction - '0') << ".txt";
#else
				ss << "CPU" << RESOLUTION_X << "x" <<RESOLUTION_Y << "Fun" << (Utils::globalFunction - '0') << ".txt";
#endif
				str = ss.str();
			}
			

			cout<<"ZAPISUJE PLIK: "<<str<<endl;

			ofstream file;
	

			file.open (str);
			file.imbue(locale("french"));
			for(int i = 0 ; i < globalToSave.rows; ++i){
				file << globalToSave.at<float>(i) << "\n";
			}
			file.close();

			if(Utils::globalFunction == '9'){
				exit(0);
			}
		}

		static inline bool collisionTest( const CvPoint2D32f& point, const int x, const int y, const int sizeX, const int sizeY ){
			if(point.x < x || point.x > x + sizeX || point.y < y || point.y > y + sizeY) return false;
			return true;
		}

		static inline bool collisionTest( int px, int py, const int x, const int y, const int sizeX, const int sizeY ){
			if(px < x-sizeX || px > x + sizeX || py < y-sizeY || py > y + sizeY) return false;
			return true;
		}

		static inline bool collisionTest( const int x, const int y, const Rect& rect){

			if(x < rect.x || x > rect.x + rect.width || y < rect.y || y > rect.y + rect.height) return false;
			return true;
		}

		/** Equation of mahalanobis distance */
		static inline double myMahalanobis(const cv::Mat& vec1, const cv::Mat& vec2, const cv::Mat& icovar){
			double sum = 0;
			for(int i = 0 ; i < vec1.rows; ++i){
				for(int j = 0 ; j < vec1.cols ; ++j){
					sum += icovar.at<double>(i,j) * (vec1.at<double>(0,i) - vec2.at<double>(0,i)) * (vec1.at<double>(0,j) * vec2.at<double>(0,j));
				}
			}
			return sum;//sqrt(sum);
		}

		static void equalizeImage(cv::Mat& frame){
			cvtColor(frame, frame, CV_BGR2HSV);

			vector<Mat> chanels;
			split(frame, chanels);

			equalizeHist(chanels[2], chanels[2]);

			merge(chanels, frame);
			cvtColor(frame, frame, CV_HSV2BGR);
		}

		static void equalizeImage(GpuFrames& gpuFrames, gpu::Stream& gpuStream){
			gpu::cvtColor(gpuFrames.frame, gpuFrames.frame, CV_BGR2HSV, 0, gpuStream);
			gpu::split(gpuFrames.frame, gpuFrames.chanels, gpuStream);
			gpu::equalizeHist(gpuFrames.chanels[2], gpuFrames.chanels[2], gpuStream);
			gpu::merge(gpuFrames.chanels, gpuFrames.frame, gpuStream);
			gpu::cvtColor(gpuFrames.frame, gpuFrames.frame, CV_HSV2BGR, 0, gpuStream);

			
		}

		static void equalizeColor( Mat &frame ) {
			Mat splitsHighlights[3];
			split(frame, splitsHighlights);
			threshold(splitsHighlights[0], splitsHighlights[0], 200, 255, THRESH_BINARY);
			threshold(splitsHighlights[1], splitsHighlights[1], 200, 255, THRESH_BINARY);
			threshold(splitsHighlights[2], splitsHighlights[2], 200, 255, THRESH_BINARY);

			multiply(splitsHighlights[0], splitsHighlights[1], splitsHighlights[1]);
			multiply(splitsHighlights[1], splitsHighlights[2], splitsHighlights[2]);

			Scalar meanVal = mean(frame, splitsHighlights[2]);
			
			double gray = (meanVal[0] + meanVal[1] + meanVal[2])/3;

			meanVal[0] = gray - meanVal[0];
			meanVal[1] = gray - meanVal[1];
			meanVal[2] = gray - meanVal[2];

			add(frame, meanVal, frame);
		}
		

		static std::string toString(double x){
			std::ostringstream o;
			
			if (!(o << x))
				return NULL;
			return o.str();
		}

		static std::string toString(int x){
			std::ostringstream o;

			if (!(o << x))
				return NULL;
			return o.str();
		}


		static double printTime( double time, const String& str ) {
			cout << str << ": " << setw (12) << (((double)getTickCount() - time)/getTickFrequency()) <<  " ";
			time = (double)getTickCount(); 	
			return time;
		}

		static void saveImg( long saveCount, const Mat& readyToHandFind ) {
			stringstream str;
			str << "trn" << saveCount << ".bmp";
			IplImage tempImg = readyToHandFind;
			cvSaveImage(str.str().c_str(), &tempImg);
			cout << "Zapisano: " << str.str() << endl;
		}


		static void binaryOpen(gpu::GpuMat& src, gpu::GpuMat& dst, gpu::GpuMat& buf, const int iterations,  gpu::Stream& gpuStream, const int structType = STRUCT_TYPE_RECT){
			cv::gpu::DevMem2D_<unsigned char> srcMem = (cv::gpu::DevMem2D_<unsigned char>)src;
			cv::gpu::DevMem2D_<unsigned char> dstMem = (cv::gpu::DevMem2D_<unsigned char>)dst;
			cv::gpu::DevMem2D_<unsigned char> bufMem = (cv::gpu::DevMem2D_<unsigned char>)buf;
			GpuFunctions::binaryOpen(srcMem, dstMem, bufMem, iterations, gpu::StreamAccessor::getStream(gpuStream), structType);
			//GpuFunctions::binaryOpen(srcMem, dstMem, bufMem, iterations, NULL, structType);
		}

		static void grayscaleOpen(gpu::GpuMat& src, gpu::GpuMat& dst, gpu::GpuMat& buf, const int iterations,  gpu::Stream& gpuStream, const int structType = STRUCT_TYPE_RECT){
			cv::gpu::DevMem2D_<unsigned char> srcMem = (cv::gpu::DevMem2D_<unsigned char>)src;
			cv::gpu::DevMem2D_<unsigned char> dstMem = (cv::gpu::DevMem2D_<unsigned char>)dst;
			cv::gpu::DevMem2D_<unsigned char> bufMem = (cv::gpu::DevMem2D_<unsigned char>)buf;
			GpuFunctions::grayscaleOpen(srcMem, dstMem, bufMem, iterations, gpu::StreamAccessor::getStream(gpuStream), structType);
			//GpuFunctions::grayscaleOpen(srcMem, dstMem, bufMem, iterations, NULL, structType);
		}

		static void grayscaleClose(gpu::GpuMat& src, gpu::GpuMat& dst, gpu::GpuMat& buf, const int iterations,  gpu::Stream& gpuStream, const int structType = STRUCT_TYPE_RECT){
			cv::gpu::DevMem2D_<unsigned char> srcMem = (cv::gpu::DevMem2D_<unsigned char>)src;
			cv::gpu::DevMem2D_<unsigned char> dstMem = (cv::gpu::DevMem2D_<unsigned char>)dst;
			cv::gpu::DevMem2D_<unsigned char> bufMem = (cv::gpu::DevMem2D_<unsigned char>)buf;
			GpuFunctions::grayscaleClose(srcMem, dstMem, bufMem, iterations, gpu::StreamAccessor::getStream(gpuStream), structType);
			//GpuFunctions::grayscaleClose(srcMem, dstMem, bufMem, iterations, NULL, structType);
		}

		static void binaryClose(gpu::GpuMat& src, gpu::GpuMat& dst, gpu::GpuMat& buf, const int iterations,  gpu::Stream& gpuStream, const int structType = STRUCT_TYPE_RECT){
			cv::gpu::DevMem2D_<unsigned char> srcMem = (cv::gpu::DevMem2D_<unsigned char>)src;
			cv::gpu::DevMem2D_<unsigned char> dstMem = (cv::gpu::DevMem2D_<unsigned char>)dst;
			cv::gpu::DevMem2D_<unsigned char> bufMem = (cv::gpu::DevMem2D_<unsigned char>)buf;
			GpuFunctions::binaryClose(srcMem, dstMem, bufMem, iterations, gpu::StreamAccessor::getStream(gpuStream), structType);
			//GpuFunctions::binaryClose(srcMem, dstMem, bufMem, iterations, NULL, structType);
		}


		static void morpholotyTimingTest( GpuFrames &gpuFrames, gpu::Stream gpuStream, Mat frame ) {
			//MORPHOLOGY TEST
			ostringstream ss;
			gpu::split(gpuFrames.afterSkinDetection, gpuFrames.chanels);



			for(int j = 0; j <= 30; j+=5){
				if(j == 0) j = 1;
				Utils::globalCounter = 0;
				for(int i = 0; i < MAX_TEST_ITERATIONS; ++ i){
					double time = (double)getTickCount();

					Utils::binaryOpen(gpuFrames.grayFrame, gpuFrames.grayFrame, gpuFrames.buf, j, gpuStream, STRUCT_TYPE_RECT);
					Utils::globalToSave.at<float>(Utils::globalCounter) = (((double)getTickCount() - time)/getTickFrequency());

					++Utils::globalCounter;
				}
				ss.str("");
				ss << gpuFrames.grayFrame.cols << "x" <<gpuFrames.grayFrame.rows << "binaryOpen" << j << ".txt";
				Utils::globalSaveToFile(ss.str());
				if(j == 1) j = 0;
			}


			for(int j = 0; j <= 30; j+=5){
				if(j == 0) j = 1;
				Utils::globalCounter = 0;
				for(int i = 0; i < MAX_TEST_ITERATIONS; ++ i){
					double time = (double)getTickCount();

					Utils::grayscaleOpen(gpuFrames.grayFrame, gpuFrames.grayFrame, gpuFrames.buf, j, gpuStream, STRUCT_TYPE_RECT);
					Utils::globalToSave.at<float>(Utils::globalCounter) = (((double)getTickCount() - time)/getTickFrequency());

					++Utils::globalCounter;
				}
				ss.str("");
				ss << gpuFrames.grayFrame.cols << "x" <<gpuFrames.grayFrame.rows << "grayScaleOpen" << j << ".txt";
				Utils::globalSaveToFile(ss.str());
				if(j == 1) j = 0;
			}




			for(int j = 0; j <= 30; j+=5){
				if(j == 0) j = 1;
				Utils::globalCounter = 0;
				for(int i = 0; i < MAX_TEST_ITERATIONS; ++ i){
					double time = (double)getTickCount();

					gpu::morphologyEx(gpuFrames.grayFrame, gpuFrames.grayFrame, CV_MOP_OPEN, Mat(), gpuFrames.buf, gpuFrames.chanels[0], Point(-1,-1), j);
					Utils::globalToSave.at<float>(Utils::globalCounter) = (((double)getTickCount() - time)/getTickFrequency());

					++Utils::globalCounter;
				}
				ss.str("");
				ss << gpuFrames.grayFrame.cols << "x" <<gpuFrames.grayFrame.rows << "gpuOpenCvOpen" << j << ".txt";
				Utils::globalSaveToFile(ss.str());
				if(j == 1) j = 0;
			}



			Mat grayFrame;
			cvtColor(frame, grayFrame, CV_BGR2GRAY);
			Mat grayFrameCopy = grayFrame.clone();


			for(int j = 0; j <= 30; j+=5){
				if(j == 0) j = 1;
				Utils::globalCounter = 0;
				for(int i = 0; i < MAX_TEST_ITERATIONS; ++ i){
					double time = (double)getTickCount();
					morphologyEx(grayFrame, grayFrame, CV_MOP_OPEN,  Mat(), Point(-1,-1), j);
					Utils::globalToSave.at<float>(Utils::globalCounter) = (((double)getTickCount() - time)/getTickFrequency());
					++Utils::globalCounter;
				}
				ss.str("");
				ss << grayFrame.cols<< "x" <<grayFrame.rows << "openCVopen" << j << ".txt";
				Utils::globalSaveToFile(ss.str());
				if(j == 1) j = 0;
			}
		}

	};
}
