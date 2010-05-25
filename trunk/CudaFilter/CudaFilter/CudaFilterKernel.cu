//------------------------------------------------------------------------------
// File: CudaFilterKernel.cu
// 
// Author: Ren Yifei, Lin Ziya
//
// Contact: yfren@cs.hku.hk, zlin@cs.hku.hk
//
// Desc: The actual CUDA code. 
// Different filter will be compiled by defining different macro.
// There're already filters below:
// 1. Sobel 2. Laplacian 3. Average 4. Highboost
//
//------------------------------------------------------------------------------

extern "C"
{
	#include "CudaFilterKernel.h"
};


#define CLAMP_8bit(x) max(0, min(255, (x)))

const int TILE_WIDTH	= 16;
const int TILE_HEIGHT	= 16;

#if	defined	COMPILE_SOBEL_FILTER
	const int FILTER_RADIUS = 1;
#elif defined COMPILE_LAPLACIAN_FILTER
	const int FILTER_RADIUS = 1;
#elif defined COMPILE_AVERAGE_FILTER
	const int FILTER_RADIUS = 3;
#elif defined COMPILE_HIGH_BOOST_FILTER
	const int FILTER_RADIUS = 3;
#endif

const int FILTER_DIAMETER = 2 * FILTER_RADIUS + 1;
const int FILTER_AREA	= FILTER_DIAMETER * FILTER_DIAMETER;

const int BLOCK_WIDTH	= TILE_WIDTH + 2 * FILTER_RADIUS;
const int BLOCK_HEIGHT	= TILE_HEIGHT + 2 * FILTER_RADIUS;

const int EDGE_VALUE_THRESHOLD = 70;
const int HIGH_BOOST_FACTOR = 10;

/* DEVICE Memory */
BYTE* d_LumaPixelsIn = NULL;
BYTE* d_LumaPixelsOut = NULL;

#ifdef COMPILE_SOBEL_FILTER

	const float h_SobelMatrix[9] = {-1,0,1,-2,0,2,-1,0,1};
	float* d_SobelMatrix;

#endif

#ifdef COMPILE_LAPLACIAN_FILTER

	const float h_LaplacianMatrix[9] = {-1,-1,-1,-1,8,-1,-1,-1,-1};
	float* d_LaplacianMatrix;

#endif

// const float h_Optimal5X5SobelMatrix[25] = 
// {
// -0.00395280095, -0.01022432803, 0.0, 0.01022432803, 0.00395280095,
// -0.02698439668, -0.069797929832, 0.0, 0.069797929832, 0.02698439668,
// -0.046510496355, -0.120304203927, 0.0, 0.120304203927, 0.046510496355,
// -0.02698439668, -0.069797929832, 0.0, 0.069797929832, 0.02698439668,
// -0.00395280095, -0.01022432803, 0.0, 0.01022432803, 0.00395280095
// };

// frame size
int* d_Width = NULL;
int* d_Height = NULL;

/* HOST Memory */
int	h_Width;
int	h_Height;
long h_DataLength;

__global__ void SobelFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, float* d_SobelMatrix);
__global__ void LaplacianFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, float* d_LaplacianMatrix);
__global__ void AverageFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height);
__global__ void HighBoostFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height);

//__global__ void MedianFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height);

void FilterWrapper(BYTE* pImageIn);


bool CUDAInit(int width, int height)
{
	//FIXME Check for device CUDA.

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
	h_DataLength = dataLength;

#ifdef COMPILE_SOBEL_FILTER

	if(d_SobelMatrix == NULL)
	{
		CUDA_SAFE_CALL( cudaMalloc((void**)&d_SobelMatrix, sizeof(float) * FILTER_AREA) );
		CUDA_SAFE_CALL( cudaMemcpy(d_SobelMatrix,  h_SobelMatrix, sizeof(float) * FILTER_AREA, cudaMemcpyHostToDevice) );
	}

#endif

#ifdef COMPILE_LAPLACIAN_FILTER

	if(d_LaplacianMatrix == NULL)
	{
		CUDA_SAFE_CALL( cudaMalloc((void**)&d_LaplacianMatrix, sizeof(float) * FILTER_AREA) );
		CUDA_SAFE_CALL( cudaMemcpy(d_LaplacianMatrix,  h_LaplacianMatrix, sizeof(float) * FILTER_AREA, cudaMemcpyHostToDevice) );
	}

#endif

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

	FilterWrapper(pImageIn);

	return true;
}

bool CUDAEndDetection(BYTE* pImageOut)
{


#ifdef COMPILE_SOBEL_FILTER
	memset(pImageOut + h_DataLength / 2, 128, h_DataLength / 2);
#elif defined COMPILE_LAPLACIAN_FILTER
	memset(pImageOut + h_DataLength / 2, 128, h_DataLength / 2);
#endif

	CUDA_SAFE_CALL( cudaMemcpy(pImageOut, d_LumaPixelsOut, sizeof(BYTE) * h_DataLength / 2, cudaMemcpyDeviceToHost) );

	return true;
}

void FilterWrapper(BYTE* pImageIn)
{
	int gridWidth  = (h_Width + TILE_WIDTH - 1) / TILE_WIDTH;
	int gridHeight = (h_Height + TILE_HEIGHT - 1) / TILE_HEIGHT;
	
	dim3 dimGrid(gridWidth, gridHeight);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);

#ifdef COMPILE_SOBEL_FILTER
	SobelFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height, d_SobelMatrix);

#elif defined COMPILE_LAPLACIAN_FILTER
	LaplacianFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height, d_LaplacianMatrix);

#elif defined COMPILE_AVERAGE_FILTER
	AverageFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height);

#elif defined COMPILE_HIGH_BOOST_FILTER
	HighBoostFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height);

#endif
	
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}


__global__ void SobelFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, float* d_SobelMatrix)
{
	__shared__ BYTE sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

	if( x < FILTER_RADIUS || x > *width  - FILTER_RADIUS - 1 || y < FILTER_RADIUS || y > *height - FILTER_RADIUS - 1)
	{
		int index = y * (*width) + x;
		g_DataOut[index] = g_DataIn[index];

		return;
	}

	//No filtering for the edges
// 	x = max(FILTER_RADIUS, x);
// 	x = min(x, *width  - FILTER_RADIUS - 1);
// 	y = max(FILTER_RADIUS, y);
// 	y = min(y, *height - FILTER_RADIUS - 1);

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

			g_DataOut[index] = abs(sumX) + abs(sumY) > EDGE_VALUE_THRESHOLD ? 255 : 0;
	}
}

__global__ void AverageFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height)
{
	__shared__ BYTE sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

	if( x < FILTER_RADIUS || x > *width  - FILTER_RADIUS - 1 || y < FILTER_RADIUS || y > *height - FILTER_RADIUS - 1)
	{
		int index = y * (*width) + x;
		g_DataOut[index] = g_DataIn[index];

		return;
	}

	//No filtering for the edges
// 	x = max(FILTER_RADIUS, x);
// 	x = min(x, *width  - FILTER_RADIUS - 1);
// 	y = max(FILTER_RADIUS, y);
// 	y = min(y, *height - FILTER_RADIUS - 1);

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

		g_DataOut[index] = BYTE(sum / FILTER_AREA);
	}	
}


__global__ void LaplacianFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height, float* d_LaplacianMatrix)
{
	__shared__ BYTE sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

	if( x < FILTER_RADIUS || x > *width  - FILTER_RADIUS - 1 || y < FILTER_RADIUS || y > *height - FILTER_RADIUS - 1)
	{
		int index = y * (*width) + x;
		g_DataOut[index] = g_DataIn[index];

		return;
	}

	//No filtering for the edges
// 	x = max(FILTER_RADIUS, x);
// 	x = min(x, *width  - FILTER_RADIUS - 1);
// 	y = max(FILTER_RADIUS, y);
// 	y = min(y, *height - FILTER_RADIUS - 1);

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
				float centerPixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
				sum += centerPixel * d_LaplacianMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
			}

			//FIXME abs?
			BYTE res = max(0, min((BYTE)sum, 255));
			g_DataOut[index] = res;
	}	
}

__global__ void HighBoostFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height)
{
	__shared__ BYTE sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

	if( x < FILTER_RADIUS || x > *width  - FILTER_RADIUS - 1 || y < FILTER_RADIUS || y > *height - FILTER_RADIUS - 1)
	{
		int index = y * (*width) + x;
		g_DataOut[index] = g_DataIn[index];

		return;
	}

	//No filtering for the edges
// 	x = max(FILTER_RADIUS, x);
// 	x = min(x, *width  - FILTER_RADIUS - 1);
// 	y = max(FILTER_RADIUS, y);
// 	y = min(y, *height - FILTER_RADIUS - 1);

	int index = y * (*width) + x;
	int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

	BYTE centerPixel = sharedMem[sharedIndex] = g_DataIn[index];

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

			g_DataOut[index] = CLAMP_8bit(centerPixel + HIGH_BOOST_FACTOR * (BYTE)(centerPixel - sum / FILTER_AREA));
}

/*__global__ void MedianFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height)
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
	}*/	
}