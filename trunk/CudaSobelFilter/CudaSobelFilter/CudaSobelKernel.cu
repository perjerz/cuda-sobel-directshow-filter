//------------------------------------------------------------------------------
// File: CudaSobelKernel.cu
// 
// Author: Ren Yifei
// 
// Desc:
//
//
//------------------------------------------------------------------------------

extern "C"
{
	#include "CudaSobelKernel.h"
	#include <stdio.h>
};

//////////////////////////////////////////////////////////////////////////
FILE* fout;
#define MYLOG(x) {fout=fopen("c:\\dbg.txt","a");fprintf(fout,"value: %d\n\n",x);fclose(fout);}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//BYTE* h_LogData = -1;

//#define LogCuda(x) {CUDA_SAFE_CALL( cudaMemcpy(h_LogData, x, sizeof(x), cudaMemcpyDeviceToHost) );\
	//MYLOG(h_LogData)\
}

typedef int TEST_TYPE;

TEST_TYPE* d_kerneltest;


//////////////////////////////////////////////////////////////////////////


const int TILE_WIDTH	= 16;
const int TILE_HEIGHT	= 16;
const int FILTER_RADIUS = 3; //  3 for averge, 1 for sobel 
const int FILTER_AREA	= (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1);
const int BLOCK_WIDTH	= TILE_WIDTH + 2 * FILTER_RADIUS;
const int BLOCK_HEIGHT	= TILE_HEIGHT + 2 * FILTER_RADIUS;

// CUDA Context

//CUcontext* sobelCtx;

/* DEVICE Memory */
BYTE* d_LumaPixelsIn = NULL;
BYTE* d_LumaPixelsOut = NULL;

// frame size
int* d_Width = NULL;
int* d_Height = NULL;


/* HOST Memory */
int	h_Width;
int	h_Height;
long h_DataLength;

__global__ void SobelFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, TEST_TYPE* d_kt);

void SobelFilterWrapper(BYTE* pImageIn);


bool CUDAInit(int width, int height)
{
	//testing !! 检查是否有CUDA设备!

	//if(cuInit(0)!=CUDA_SUCCESS)
		//return false;

	//cuDeviceGet(&cuDevice, 0);
	//cuCtxCreate(&cuCtx, 0, cuDevice);


	h_Width = width;
	h_Height = height;

// 	MYLOG(h_Width);
// 	MYLOG(h_Height);

	return true;
}

void CUDARelease()
{
	CUDA_SAFE_CALL( cudaFree(d_LumaPixelsIn) );
	CUDA_SAFE_CALL( cudaFree(d_LumaPixelsOut) );
	CUDA_SAFE_CALL( cudaFree(d_Width) );
	CUDA_SAFE_CALL( cudaFree(d_Height) );
}

bool CUDABeginDetection(BYTE* pImageIn, long dataLength)
{
	//////////////////////////////////////////////////////////////////////////
	CUDA_SAFE_CALL( cudaMalloc(&d_kerneltest, sizeof(TEST_TYPE)) );
	//////////////////////////////////////////////////////////////////////////


	h_DataLength = dataLength;


	if(d_Width == NULL && d_Height == NULL)
	{
		CUDA_SAFE_CALL( cudaMalloc(&d_Width, sizeof(int)) );
		CUDA_SAFE_CALL( cudaMalloc(&d_Height, sizeof(int)) );

		CUDA_SAFE_CALL( cudaMemcpy(d_Width,  &h_Width, sizeof(int), cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy(d_Height, &h_Height, sizeof(int), cudaMemcpyHostToDevice) );
	}

	if(d_LumaPixelsIn == NULL)
	{
		CUDA_SAFE_CALL( cudaMalloc((void**)&d_LumaPixelsIn, sizeof(BYTE) * h_DataLength / 2) );
	}

	if(d_LumaPixelsOut == NULL)
	{
		CUDA_SAFE_CALL( cudaMalloc((void**)&d_LumaPixelsOut, sizeof(BYTE) * h_DataLength / 2) );
	}

	CUDA_SAFE_CALL( cudaMemcpy((void*)d_LumaPixelsIn, (void*)pImageIn, sizeof(BYTE) * h_DataLength / 2, cudaMemcpyHostToDevice) );

	SobelFilterWrapper(pImageIn);

	return true;
}

bool CUDAEndDetection(BYTE* pImageOut)
{
	CUDA_SAFE_CALL( cudaMemcpy(pImageOut, d_LumaPixelsOut, sizeof(BYTE) * h_DataLength / 2, cudaMemcpyDeviceToHost) );

	return true;
}

void SobelFilterWrapper(BYTE* pImageIn)
{
	int gridWidth = (h_Width + TILE_WIDTH - 1) / TILE_WIDTH;
	int gridHeight = (h_Height + TILE_HEIGHT - 1) / TILE_HEIGHT;
	
	dim3 dimGrid(gridWidth, gridHeight);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);

	SobelFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height, d_kerneltest);	

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


__global__ void SobelFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, TEST_TYPE* d_kt)
{
	__shared__ BYTE sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

	//No filtering for the edges
	x = max(FILTER_RADIUS, x);
	x = min(x, *width  - FILTER_RADIUS - 1);
	y = max(FILTER_RADIUS, y);
	y = min(y, *height - FILTER_RADIUS - 1);

	int index = y * (*width) + x;
	int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

	sharedMem[sharedIndex] = g_DataIn[index];

	__syncthreads();

	if(		threadIdx.x >= FILTER_RADIUS && threadIdx.x < BLOCK_WIDTH - FILTER_RADIUS 
		&&	threadIdx.y >= FILTER_RADIUS && threadIdx.y < BLOCK_HEIGHT - FILTER_RADIUS)
	{
		float sum = 0;

		for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
		for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
		{
			float pixelValue = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);

			sum += pixelValue;
		}

		BYTE res = (BYTE)(sum / FILTER_AREA);

		g_DataOut[index] = res;
	}	
}