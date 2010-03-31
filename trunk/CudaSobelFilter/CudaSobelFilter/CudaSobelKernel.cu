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

const int FILTER_RADIUS = 1; //  3 for averge, 1 for sobel
const int FILTER_DIAMETER = 2 * FILTER_RADIUS + 1;
const int FILTER_AREA	= FILTER_DIAMETER * FILTER_DIAMETER;

const int BLOCK_WIDTH	= TILE_WIDTH + 2 * FILTER_RADIUS;
const int BLOCK_HEIGHT	= TILE_HEIGHT + 2 * FILTER_RADIUS;

const int SOBEL_THRESHOLD = 50;

/* DEVICE Memory */
 BYTE* d_LumaPixelsIn = NULL;
 BYTE* d_LumaPixelsOut = NULL;

// Sobel matrix
int* d_SobelMatrix;

int h_SobelMatrix[9] = {-1,0,1,-2,0,2,-1,0,1};

// frame size
 int* d_Width = NULL;
 int* d_Height = NULL;


/* HOST Memory */
int	h_Width;
int	h_Height;
long h_DataLength;

__global__ void SobelFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, int* d_SobelMatrix, TEST_TYPE* d_kt);
__global__ void AverageFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, TEST_TYPE* d_kt);
__global__ void MedianFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, TEST_TYPE* d_kt);

void SobelFilterWrapper(BYTE* pImageIn);


bool CUDAInit(int width, int height)
{
	//testing !! 检查是否有CUDA设备!


	h_Width = width;
	h_Height = height;

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

	if(d_SobelMatrix == NULL)
	{
		CUDA_SAFE_CALL( cudaMalloc((void**)&d_SobelMatrix, sizeof(int) * 9) );
		CUDA_SAFE_CALL( cudaMemcpy(d_SobelMatrix,  h_SobelMatrix, sizeof(int) * 9, cudaMemcpyHostToDevice) );
	}

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

	SobelFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height, d_SobelMatrix, d_kerneltest);	

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

__global__ void SobelFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, int* d_SobelMatrix, TEST_TYPE* d_kt)
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
		float sumX = 0, sumY=0;

		for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
			for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
			{
				float centerPixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
				sumX += centerPixel * d_SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
				sumY += centerPixel * d_SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy+FILTER_RADIUS)];
			}

			g_DataOut[index] = abs(sumX) + abs(sumY) > SOBEL_THRESHOLD ? 128 : 0;
	}	
}

__global__ void AverageFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, TEST_TYPE* d_kt)
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



__global__ void MedianFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, TEST_TYPE* d_kt)
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

	//可以用256个桶，然后放，最后取得排第128的

	BYTE sortCuda[256]; for(int i=0;i<256;++i) sortCuda[i]=0;
	
	if(		threadIdx.x >= FILTER_RADIUS && threadIdx.x < BLOCK_WIDTH - FILTER_RADIUS 
		&&	threadIdx.y >= FILTER_RADIUS && threadIdx.y < BLOCK_HEIGHT - FILTER_RADIUS)
	{
		//float sum = 0;

		for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
			for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
			{
				sortCuda[(sharedMem[sharedIndex + (dy * blockDim.x + dx)])] += 1;

				//sum += pixelValue;
			}

			//BYTE res = (BYTE)(sum / FILTER_AREA);

			int cnt=0;
			int res=0;
			for(int i=0; i<256; ++i)
				if(cnt>=127)
				{
					res = i;
					break;
				}
				else
				{
					cnt+=sortCuda[i];
				}

			g_DataOut[index] = res;
	}	
}