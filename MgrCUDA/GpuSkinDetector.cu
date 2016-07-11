#include "GpuSkinDetector.h"


#include <cutil.h>

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


/////////////////////////////////////////// EMPTY //////////////////////////////////////////////////////////////////

__global__ void gpuEmptyFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];
		if(px.r != 0 && px.g != 0 && px.b != 0){
			Pixel white;
			white.r = 255;
			white.g = 255;
			white.b = 255;
			dst.ptr(y)[x] = white;
		}else{
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			dst.ptr(y)[x] = black;
		}
	}
}

/////////////////////////////////////////// yCbCr //////////////////////////////////////////////////////////////////

static const int range_cb_min = 110;
static const int range_cb_max = 141;
static const int range_cr_min = 128;//130
static const int range_cr_max= 155;

#ifndef FIT_RANGE
#define FIT_RANGE(v,l,h) (MIN(h, MAX(l, v)))
#endif

__global__ void gpuYCbCrFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];
		float r = px.r;
		float g = px.g;
		float b = px.b;

		//return true;
		// convert to YCbCr 
		//y  =  0.2989f*r + 0.5866f*g + 0.1145f*b + 0.5f;
		float cb = -0.1687f*r - 0.3312f*g + 0.500f*b  + 128.0f;
		float cr =  0.500f*r  - 0.4183f*g - 0.0816f*b + 128.0f;

		FIT_RANGE(cb, 0.0, 255.0);
		FIT_RANGE(cr, 0.0, 255.0);
		// end convert 

		float sum = r+g+b;

		if (
			r>95&&g> 40 && b > 20 
			&& MAX(MAX(r,g),b)-MIN(MIN(r,g),b) > 15 
			&& abs(r-g) > 15 && r > g && r > b 
			&& (b/g<1.249)&& b/g > 0.5 &&
			(sum/(3*r)>0.692)&&//0.696
			(0.3333-b/sum>0.029)&&//0.014
			(g/(3*sum)<0.124)

			||	(3*b*r*r)/(sum*sum*sum) >0.110  //0.1276
			&& ((r*b + g*g ) / g*b) > 5000 
			&& sum/(3*r + (r-g)/sum) < 2.7775 //2.7775 lub czasem 1.06
			&& cb >= range_cb_min 
			&& cb <= range_cb_max
			&& cr >= range_cr_min 
			&& cr <= range_cr_max
			){
			Pixel white;
			white.r = 255;
			white.g = 255;
			white.b = 255;
			dst.ptr(y)[x] = white;
		}else{
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			dst.ptr(y)[x] = black;
		}
	}
}

/////////////////////////////////////////// GAUSSIAN //////////////////////////////////////////////////////////////////


__global__ void gpuGausianFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];
		float sum = px.r + px.g + px.b;
		
		if ((px.g/px.b - px.r/px.g<=-0.0905)&&
			((px.g*sum)/(px.b*(px.r-px.g))>3.4857)&&
			((sum*sum*sum)/(3*px.g*px.r*px.r)<=7.397)&&
			(sum/(9*px.r)-0.333 > -0.0976)){

			Pixel white;
			white.r = 255;
			white.g = 255;
			white.b = 255;
			dst.ptr(y)[x] = white;
		}else{
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			dst.ptr(y)[x] = black;
		}
	}
}

/////////////////////////////////////////// HSV //////////////////////////////////////////////////////////////////


__global__ void gpuHsvFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];

		/* convert to HSV */
		float tc = 0;

		float r_float;
		float g_float;
		float b_float;

		float h;
		float s;
		float v;

		r_float = ((float)px.r)/255.0;
		g_float = ((float)px.g)/255.0;
		b_float = ((float)px.b)/255.0;

		v = MAX(px.r,MAX(px.g,px.b));
		s = (v == 0 ? 0 : ((v - MIN(px.r,MIN(px.g,px.b))) / v));

		if (v == r_float){
			h =       (g_float - b_float) * 60 / s;
		} else if (v == g_float) {
			h = 180 + (b_float - r_float) * 60 / s;
		} else { 
			h = 240 + (r_float - g_float) * 60 / s;
		}

		if (h < 0){
			h += 360;
		}

		s *= 255.0;
		v *= 255.0;
		/* end convert */

		int range_h_min;
		int range_h_max;
		int range_s_min;
		int range_s_max;


		range_h_min = 244;
		range_h_max = 267;
		range_s_min = 87;
		range_s_max = 200;

		float sum = px.r+px.b+px.g;
		if (        
			h >= range_h_min && 
			h <= range_h_max &&      
			s >= range_s_min && 
			s <= range_s_max
			&&px.r>95&&px.g> 40 && px.b > 20 
			&& (px.b/px.g<1.249)&& px.b/px.g > 0.5 &&
			(sum/(3*px.r)>tc/0.692)&&
			(0.3333-px.b/sum>0.029)&&
			(px.g/(3*sum)<0.124)

			||	(3*px.b*px.r*px.r)/(sum*sum*sum) >0.110 && 
			((px.r*px.b + px.g*px.g ) / px.g*px.b) > 5000 &&
			sum/(3*px.r + (px.r-px.g)/sum) <2.7775 


			){

			Pixel white;
			white.r = 255;
			white.g = 255;
			white.b = 255;
			dst.ptr(y)[x] = white;
		}else{
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			dst.ptr(y)[x] = black;
		}
	}
}

/////////////////////////////////////////// SIMPLE //////////////////////////////////////////////////////////////////


__global__ void gpuSimpleFunction(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];
		float r = px.r;
		float g = px.g;
		float b = px.b;
		
		float sum = r+g+b;

		float a1 = r/b;
		float a2 = (r*b)/(sum*sum);
		float a3 = (r*g)/(sum*sum);

		if ( a1 > 1.185 && a2 > 0.107 && a3 > 0.112){
			Pixel white;
			white.r = 255;
			white.g = 255;
			white.b = 255;
			dst.ptr(y)[x] = white;
		}else{
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			dst.ptr(y)[x] = black;
		}
	}
}

__global__ void gpuSimpleFunction2(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];

		float r = px.r;
		float g = px.g;
		float b = px.b;
		
		float sum = r+g+b;

		if (
			r > 95 && g > 40 && b > 20 
			&& MAX(MAX(r,g),b) - MIN(MIN(r,g),b) > 15 
			&& abs(r-g) > 15 
			&& r > g 
			&& r > b 
			&& (b/g < 1.249)
			&& b/g > 0.5 
			&& (sum/(3*r) > 0.692) 
			&& (0.3333-b/sum > 0.029)
			&& (g/(3*sum) < 0.124) || (3*b*r*r)/(sum*sum*sum) > 0.110 
			&& ((r*b + g*g ) / g*b) > 5000 
			&& sum/(3*r + (r-g)/sum) <2.7775 //2.7775 lub czasem 1.06
			){
			Pixel white;
			white.r = 255;
			white.g = 255;
			white.b = 255;
			dst.ptr(y)[x] = white;
		}else{
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			dst.ptr(y)[x] = black;
		}
	}
}
__global__ void gpuSimpleFunction3(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];
		float r = px.r;
		float g = px.g;
		float b = px.b;

		float sum = r+g+b;

		if ((g / b - r / b <= -0.0905) && (sum / (3*r) + (r- g) / sum <= 0.9498)){
			Pixel white;
			white.r = 255;
			white.g = 255;
			white.b = 255;
			dst.ptr(y)[x] = white;
		}else{
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			dst.ptr(y)[x] = black;
		}
	}
}
__global__ void gpuSimpleFunction4(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];
		float r = px.r;
		float g = px.g;
		float b = px.b;

		float sum = r+g+b;

		if((b/g < 1.249)
			&& (sum/(3*r) > 0.692)
			&& (0.3333-b/sum > 0.029)
			&& (g/(3*sum) < 0.124)){

			Pixel white;
			white.r = 255;
			white.g = 255;
			white.b = 255;
			dst.ptr(y)[x] = white;
		}else{
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			dst.ptr(y)[x] = black;
		}
	}
}
__global__ void gpuSimpleFunction5(const cv::gpu::DevMem2D_<Pixel> src, cv::gpu::DevMem2D_<Pixel> dst) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];
		float r = px.r;
		float g = px.g;
		float b = px.b;

		float sum = r+g+b;

		if(g/b - r/g<=-0.0905
			&& (g*sum)/(b*(r-g)) > 3.4857
			&& (sum*sum*sum)/(3*g*r*r) <= 7.397
			&& sum/(9*r)-0.333 > -0.0976){
			Pixel white;
			white.r = 255;
			white.g = 255;
			white.b = 255;
			dst.ptr(y)[x] = white;
		}else{
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			dst.ptr(y)[x] = black;
		}
	}
}

__global__ void gpuInitialFilter(cv::gpu::DevMem2D_<Pixel> src) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src.cols && y < src.rows){
		Pixel px = src.ptr(y)[x];
		float r = px.r;
		float g = px.g;
		float b = px.b;

		float sum = r + g + b;
		if( (b > 160 && r < 180 && g < 180) || //Too much blue
			(g > 160 && r < 180 && b < 180) || //Too much green
			(b < 70 && r < 70 && g < 70) || //Too dark
			//(g > 200 && b < 80 && r < 100) || //Green
			(r+g > 400 && b < 170) || // Too much red and gree ( yellow like colour)
			//(g > 110 && b < 90) || //Yellow like also (bylo 150)
			(b/(sum) > .4) || //Too much blue in contrast to others
			(g/(sum) > .4)  //Too much green in contrast to others
			//(r < 102 && g > 100 && b > 110 && g < 140 && b < 160)|| //Ocean
			//(r>240 && g> 230 && b>230)
			//||(r==255&&g==255&&b==255)//eliminate white

			|| (r<240 && g<240 && b<240 && abs(r-g)<10 && abs(g-b)<20 && abs(r-b) < 30 && r>200)//sprawdzic czy 20-30 to nie za duzo
			|| (r<240 && g<240 && b<240 && abs(r-b)<5)
		){
			Pixel black;
			black.r = 0;
			black.g = 0;
			black.b = 0;
			src.ptr(y)[x] = black;
		}
	}
}

void GpuSkinDetector::initialFilter( cv::gpu::DevMem2D_<Pixel>& src, cudaStream_t gpuStream){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);
	gpuInitialFilter<<<grid, block, 0, gpuStream>>>(src);
			
	if( gpuStream == 0 ){
		cudaDeviceSynchronize();
	}
}

void GpuSkinDetector::detectSkin( const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, FUNCTION function, cudaStream_t gpuStream){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);
	
	switch(function){
		case EMPTY:{
			gpuEmptyFunction<<<grid, block, 0, gpuStream>>>(src, dst);
			break;
		}
		case YCBCR:{
			gpuYCbCrFunction<<<grid, block, 0, gpuStream>>>(src, dst);
			break;
		}
		case HSV:{
			gpuHsvFunction<<<grid, block, 0, gpuStream>>>(src, dst);
			break;
		}
		case GAUSSIAN:{
			gpuGausianFunction<<<grid, block, 0, gpuStream>>>(src, dst);
			break;
		}
		case SIMPLE:{
			gpuSimpleFunction<<<grid, block, 0, gpuStream>>>(src, dst);
			break;
		}

		case SIMPLE2:{
			gpuSimpleFunction2<<<grid, block, 0, gpuStream>>>(src, dst);
			break;
		}
		case SIMPLE3:{
			gpuSimpleFunction3<<<grid, block, 0, gpuStream>>>(src, dst);
			break;
		}
		case SIMPLE4:{
			gpuSimpleFunction4<<<grid, block, 0, gpuStream>>>(src, dst);
			break;
		}
		case SIMPLE5 :{
			gpuSimpleFunction5<<<grid, block, 0, gpuStream>>>(src, dst);
			break;
		}
	}
	
	
	//cudaSafeCall( cudaGetLastError() );

    if (gpuStream == 0) {
		cudaDeviceSynchronize();
	}
}



void GpuSkinDetector::yCbCrFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ){
	detectSkin(src, dst, YCBCR, gpuStream);
}

void GpuSkinDetector::emptyFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ) {
	detectSkin(src,dst, EMPTY, gpuStream);
}

void GpuSkinDetector::gausianFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ) {
	detectSkin(src,dst, GAUSSIAN, gpuStream);
}

void GpuSkinDetector::hsvFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ) {
	detectSkin(src,dst, HSV, gpuStream);
}

void GpuSkinDetector::simpleFunction(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ) {
	detectSkin(src,dst, SIMPLE, gpuStream);
}

void GpuSkinDetector::simpleFunction2(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ) {
	detectSkin(src,dst, SIMPLE2, gpuStream);
}

void GpuSkinDetector::simpleFunction3(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ) {
	detectSkin(src,dst, SIMPLE3, gpuStream);
}

void GpuSkinDetector::simpleFunction4(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ) {
	detectSkin(src,dst, SIMPLE4, gpuStream);
}

void GpuSkinDetector::simpleFunction5(const cv::gpu::DevMem2D_<Pixel>& src, cv::gpu::DevMem2D_<Pixel>& dst, cudaStream_t gpuStream ) {
	detectSkin(src,dst, SIMPLE5, gpuStream);
}




