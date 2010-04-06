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

// Decide which filter to compile

//////////////////////////////////////////////////////////////////////////

//#define COMPILE_SOBEL_FILTER
//#define COMPILE_LAPLACIAN_FILTER
#define COMPILE_AVERAGE_FILTER
//#define COMPILE_HIGH_BOOST_FILTER

//////////////////////////////////////////////////////////////////////////


bool CUDAInit(int width, int height);

void CUDARelease();

bool CUDABeginDetection(BYTE* pImageIn, long dataLength);

bool CUDAEndDetection(BYTE* pImageOut);

#endif