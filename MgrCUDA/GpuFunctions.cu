#include "GpuFunctions.h"
#include <cutil.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// host code
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gpuBinaryErrode(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations, const int morphType) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < src.cols && y < src.rows){
		//unsigned char p = 255;
		int iterationKw = iterations * iterations;

		for(int i = -iterations; i <= iterations; ++i){
			for(int j = -iterations; j <= iterations; ++j){
				if((morphType != STRUCT_TYPE_CIRCLE || i*i + j*j < iterationKw) && x + i > 0 && x + i < src.cols && y + j > 0 && y + j < src.rows){
					if(src.ptr(y+j)[x+i] != 255){
						dst.ptr(y)[x] = 0;
						return;
					}
					//p &= src.ptr(y+j)[x+i];
				}
			}
		}
		//dst.ptr(y)[x] = p;
		dst.ptr(y)[x] = 255;
	}
}





__global__ void gpuFastBinaryErrodeStep1(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < src.cols && y < src.rows){
		for(int i = -iterations; i <= iterations; ++i){
			if(x + i > 0 && x + i < src.cols){
				if(src.ptr(y)[x+i] != 255){
					dst.ptr(y)[x] = 0;
					return;
				}
			}
		}
		dst.ptr(y)[x] = 255;
	}
}
__global__ void gpuFastBinaryErrodeStep2(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < src.cols && y < src.rows){
		for(int i = -iterations; i <= iterations; ++i){
			if(y + i > 0 && y + i < src.rows){
				if(src.ptr(y+i)[x] != 255){
					dst.ptr(y)[x] = 0;
					return;
				}
			}
		}
		dst.ptr(y)[x] = 255;
	}
}








__global__ void gpuGrayscaleErrode(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations, const int morphType) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < src.cols && y < src.rows){
		unsigned char p = 255;
		int iterationKw = iterations * iterations;

		for(int i = -iterations; i <= iterations; ++i){
			for(int j = -iterations; j <= iterations; ++j){
				if((morphType != STRUCT_TYPE_CIRCLE || i*i + j*j < iterationKw) && x + i > 0 && x + i < src.cols && y + j > 0 && y + j < src.rows){
					p = (src.ptr(y+j)[x+i] < p) ? src.ptr(y+j)[x+i] : p;
				}
			}
		}
		dst.ptr(y)[x] = p;
	}
}

__global__ void gpuBinaryDilate(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations, const int morphType) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < src.cols && y < src.rows){
		if(src.ptr(y)[x] == 0 ){
			//unsigned char p = 0;
			int iterationKw = iterations * iterations;

			for(int i = -iterations; i <= iterations; ++i){
				for(int j = -iterations; j <= iterations; ++j){
					if((morphType != STRUCT_TYPE_CIRCLE || i*i + j*j < iterationKw) && x + i > 0 && x + i < src.cols && y + j > 0 && y + j < src.rows){
						//p |= src.ptr(y+j)[x+i];
						if(src.ptr(y+j)[x+i] == 255){
							dst.ptr(y)[x] = 255;
							return;
						}
					}
				}
			}
			dst.ptr(y)[x] = 0;
			//dst.ptr(y)[x] = p;
		}else{
			dst.ptr(y)[x] = 255;
		}
	}
}

__global__ void gpuFastBinaryDilateStep1(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < src.cols && y < src.rows){
		if(src.ptr(y)[x] == 0 ){

			for(int i = -iterations; i <= iterations; ++i){
				if( x + i > 0 && x + i < src.cols){
					if(src.ptr(y)[x+i] == 255){
						dst.ptr(y)[x] = 255;
						return;
					}
				}
			}
			dst.ptr(y)[x] = 0;
		}else{
			dst.ptr(y)[x] = 255;
		}
	}
}

__global__ void gpuFastBinaryDilateStep2(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < src.cols && y < src.rows){
		if(src.ptr(y)[x] == 0 ){
			for(int i = -iterations; i <= iterations; ++i){
				if( y + i > 0 && y + i < src.rows){
					if(src.ptr(y+i)[x] == 255){
						dst.ptr(y)[x] = 255;
						return;
					}
				}
			}
			dst.ptr(y)[x] = 0;
		}else{
			dst.ptr(y)[x] = 255;
		}
	}
}





__global__ void gpuGrayscaleDilate(const cv::gpu::DevMem2D_<unsigned char> src, cv::gpu::DevMem2D_<unsigned char> dst, const int iterations, const int morphType) { 
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < src.cols && y < src.rows){
		unsigned char p = 0;
		int iterationKw = iterations * iterations;

		for(int i = -iterations; i <= iterations; ++i){
			for(int j = -iterations; j <= iterations; ++j){
				if((morphType != STRUCT_TYPE_CIRCLE || i*i + j*j < iterationKw) && x + i > 0 && x + i < src.cols && y + j > 0 && y + j < src.rows){
					p = (src.ptr(y+j)[x+i] > p) ? src.ptr(y+j)[x+i] : p;
				}
			}
		}
		dst.ptr(y)[x] = p;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// client code
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void GpuFunctions::binaryErrode(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, const int iterations, cudaStream_t gpuStream, const int morphType){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);
	if(morphType == STRUCT_TYPE_RECT){
		gpuFastBinaryErrodeStep1<<<grid, block, 0, gpuStream>>>(src, dst, iterations);
		gpuFastBinaryErrodeStep2<<<grid, block, 0, gpuStream>>>(src, dst, iterations);
	}else{
		gpuBinaryErrode<<<grid, block, 0, gpuStream>>>(src, dst, iterations, morphType);
	}

	
	if( gpuStream == 0 ){
		cudaDeviceSynchronize();
	}
}

void GpuFunctions::grayscaleErrode(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, const int iterations, cudaStream_t gpuStream, const int morphType){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);
	gpuGrayscaleErrode<<<grid, block, 0, gpuStream>>>(src, dst, iterations, morphType);
			
	if( gpuStream == 0 ){
		cudaDeviceSynchronize();
	}
}

void GpuFunctions::binaryDilate(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, const int iterations, cudaStream_t gpuStream, const int morphType){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);

	if(morphType == STRUCT_TYPE_RECT){
		gpuFastBinaryDilateStep1<<<grid, block, 0, gpuStream>>>(src, dst, iterations);
		gpuFastBinaryDilateStep2<<<grid, block, 0, gpuStream>>>(src, dst, iterations);
	}else{
		gpuBinaryDilate<<<grid, block, 0, gpuStream>>>(src, dst, iterations, morphType);
	}

	
			
	if( gpuStream == 0 ){
		cudaDeviceSynchronize();
	}
}

void GpuFunctions::grayscaleDilate(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, const int iterations, cudaStream_t gpuStream, const int morphType){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);
	gpuGrayscaleDilate<<<grid, block, 0, gpuStream>>>(src, dst, iterations, morphType);
			
	if( gpuStream == 0 ){
		cudaDeviceSynchronize();
	}
}

void GpuFunctions::binaryOpen(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, cv::gpu::DevMem2D_<unsigned char>& buffor,  const int iterations,  cudaStream_t gpuStream, const int morphType){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);

	if(morphType == STRUCT_TYPE_RECT){
		gpuFastBinaryErrodeStep1<<<grid, block, 0, gpuStream>>>(src, buffor, iterations);
		gpuFastBinaryErrodeStep2<<<grid, block, 0, gpuStream>>>(buffor, dst, iterations);

		gpuFastBinaryDilateStep1<<<grid, block, 0, gpuStream>>>(dst, buffor, iterations);
		gpuFastBinaryDilateStep2<<<grid, block, 0, gpuStream>>>(buffor, dst, iterations);
	}else{
		gpuBinaryErrode<<<grid, block, 0, gpuStream>>>(src, buffor, iterations, morphType);
		gpuBinaryDilate<<<grid, block, 0, gpuStream>>>(buffor, dst, iterations, morphType);
	}

	if( gpuStream == 0 ){
		cudaDeviceSynchronize();
	}
}

void GpuFunctions::grayscaleOpen(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, cv::gpu::DevMem2D_<unsigned char>& buffor,  const int iterations,  cudaStream_t gpuStream, const int morphType){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);
	gpuGrayscaleErrode<<<grid, block, 0, gpuStream>>>(src, buffor, iterations, morphType);
	gpuGrayscaleDilate<<<grid, block, 0, gpuStream>>>(buffor, dst, iterations, morphType);	
			
	if( gpuStream == 0 ){
		cudaDeviceSynchronize();
	}
}

void GpuFunctions::binaryClose(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, cv::gpu::DevMem2D_<unsigned char>& buffor, const int iterations, cudaStream_t gpuStream, const int morphType){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);

	if(morphType == STRUCT_TYPE_RECT){
		gpuFastBinaryDilateStep1<<<grid, block, 0, gpuStream>>>(src, buffor, iterations);
		gpuFastBinaryDilateStep2<<<grid, block, 0, gpuStream>>>(buffor, dst, iterations);

		gpuFastBinaryErrodeStep1<<<grid, block, 0, gpuStream>>>(dst, buffor, iterations);
		gpuFastBinaryErrodeStep2<<<grid, block, 0, gpuStream>>>(buffor, dst, iterations);
	}else{
		gpuBinaryDilate<<<grid, block, 0, gpuStream>>>(src, buffor, iterations, morphType);
		gpuBinaryErrode<<<grid, block, 0, gpuStream>>>(buffor, dst, iterations, morphType);
	}
			
	if( gpuStream == 0 ){
		cudaDeviceSynchronize();
	}
}

void GpuFunctions::grayscaleClose(const cv::gpu::DevMem2D_<unsigned char>& src, cv::gpu::DevMem2D_<unsigned char>& dst, cv::gpu::DevMem2D_<unsigned char>& buffor, const int iterations, cudaStream_t gpuStream, const int morphType){
	dim3 block(16, 16);

	int grida = src.cols / block.x + !!(src.cols % block.x);
	int gridb = src.rows / block.y + !!(src.rows % block.y);

    dim3 grid(grida, gridb);
	gpuGrayscaleDilate<<<grid, block, 0, gpuStream>>>(src, buffor, iterations, morphType);
	gpuGrayscaleErrode<<<grid, block, 0, gpuStream>>>(buffor, dst, iterations, morphType);
			
	if( gpuStream == 0 ){
		cudaDeviceSynchronize();
	}
}