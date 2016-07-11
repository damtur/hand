#include "stdafx.h"

void HandGR::SkinDetectionFunction::detectSkin( const cv::Mat& before, cv::Mat& after, uchar skinColor /*= 255*/, uchar nonSkinColor /*= 0*/ ){
	//after = after.zeros(after.rows, after.cols, after.type());
	
	for(int i = 0 ; i < before.rows; ++i){
		for (int j = 0; j < before.cols; ++j){
			float b = before.at<cv::Vec3b >(i,j)[0];
			float g = before.at<cv::Vec3b >(i,j)[1];
			float r = before.at<cv::Vec3b >(i,j)[2];
			if(r==g && g==b && b==0){
				after.at<cv::Vec3b>(i,j)[0] = after.at<cv::Vec3b>(i,j)[1] = after.at<cv::Vec3b>(i,j)[2] = nonSkinColor;
				continue;
			}
			
			if( initialFilterEnabled ? isSkinInitialFilter(r,g,b) && isSkin(r,g,b) : isSkin(r,g,b) ){
				after.at<cv::Vec3b>(i,j)[0] = after.at<cv::Vec3b>(i,j)[1] = after.at<cv::Vec3b>(i,j)[2] = skinColor;
			}else{
				after.at<cv::Vec3b>(i,j)[0] = after.at<cv::Vec3b>(i,j)[1] = after.at<cv::Vec3b>(i,j)[2] = nonSkinColor;
			}
		}
	}
}


void HandGR::SkinDetectionFunction::detectSkin( const gpu::GpuMat& afterProcessingImage, gpu::GpuMat& afterSkinDetection, gpu::Stream& gpuStream){
	gpu::DevMem2D_<Pixel> src = (cv::gpu::DevMem2D_<Pixel>)afterProcessingImage;
	gpu::DevMem2D_<Pixel> dst = (cv::gpu::DevMem2D_<Pixel>)afterSkinDetection;


	cudaStream_t cudaStream = gpu::StreamAccessor::getStream(gpuStream);

//double time = (double)getTickCount();

	if( initialFilterEnabled){
		GpuSkinDetector::initialFilter(src, cudaStream);
	}

	gpuIsSkin(src, dst, cudaStream);

//
//if(Utils::globalCounter>=MAX_TEST_ITERATIONS){
//	//Utils::globalSaveToFile();
//}else{
//	Utils::globalToSave.at<float>(Utils::globalCounter) = (((double)getTickCount() - time)/getTickFrequency());
//}
//
//ostringstream ss1;
//ss1 << (Utils::globalFunction - '0')<< "_" << Utils::globalCounter<<"_";
//Utils::printTime(time, ss1.str());
//cout<<endl;
}


bool inline HandGR::SkinDetectionFunction::isSkinInitialFilter( float r, float g, float b )
{

	float sum = r + g + b;
	if( (b > 160 && r < 180 && g < 180) || //Too much blue
		(g > 160 && r < 180 && b < 180) || //Too much green
		(b < 70 && r < 70 && g < 70) || //Too dark
		//(g > 200 && b < 80 && r < 100) || //Green
		(r+g > 400 && b < 170) || // Too much red and gree ( yellow like color)
		//(g > 110 && b < 90) || //Yellow like also (bylo 150)
		(b/(sum) > .4) || //To much blue in contrast to others
		(g/(sum) > .4)  //To much green in contrast to others
		//(r < 102 && g > 100 && b > 110 && g < 140 && b < 160)|| //Ocean
		//(r>240 && g> 230 && b>230)
		//||(r==255&&g==255&&b==255)//eliminate white
		
		|| (r<240 && g<240 && b<240 && abs(r-g)<10 && abs(g-b)<20 && abs(r-b) < 30 && r>200)//sprawdzic czy 20-30 to nie za duzo
		|| (r<240 && g<240 && b<240 && abs(r-b)<5)
		){


//  	if( (b > 160 && r < 180 && g < 180))
//  		return false;
//   	if( //Too much blue
//   		(g > 160 && r < 180 && b < 180))
//  		return false;
//  	if( //Too much green
// 		(b < 70 && r < 70 && g < 70) )
// 		return false;
// 	if( //Too dark
// 		g > 200 && b < 80 && r < 100)
// 		return false;
//   	if( //Green
//   		(r+g > 400) && b < 170)
//   		return false;
// 
//  	if( // Too much red and gree ( yellow like color)
// 		(g > 110 && b < 80) )
// 		return false;
//  	if( //Yellow like also (bylo 150)
//  	(b/(sum) > .4) )
//  		return false;
//  	if( //To much blue in contrast to others
// 		(g/(sum) > .4) )
// 		return false;
// 	if( //To much green in contrast to others slaby 
// 		(r < 102 && g > 100 && b > 110 && g < 140 && b < 160) //Ocean
//  		){

			return false;//not skin pixel
	}
	return true;//maybe skin piskel
}
