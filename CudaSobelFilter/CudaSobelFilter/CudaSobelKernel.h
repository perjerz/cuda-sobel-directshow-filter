#ifndef _CUDA_SOBEL_KERNEL_H_
#define _CUDA_SOBEL_KERNEL_H_

#include <cuda.h>

typedef unsigned char       BYTE;


bool CUDAInit(unsigned int width, unsigned height);
void CUDARelease();

bool CUDABeginDetection(BYTE* pImageIn, long dataLength);

bool CUDAEndDetection(BYTE* pImageOut);

//void SobelFilterWrapper(int* g_DataIn, int* g_DataOut, unsigned int width, unsigned int height);

#endif