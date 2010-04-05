//------------------------------------------------------------------------------
// File: CudaSobelKernel.h
// 
// Author: Ren Yifei
// 
// Desc:
//
//
//------------------------------------------------------------------------------

#ifndef _CUDA_SOBEL_KERNEL_H_
#define _CUDA_SOBEL_KERNEL_H_

#include <cuda.h>
#include <cutil.h>


typedef unsigned char BYTE;

bool CUDAInit(int width, int height);

void CUDARelease();

bool CUDABeginDetection(BYTE* pImageIn, long dataLength);

bool CUDAEndDetection(BYTE* pImageOut);

#endif