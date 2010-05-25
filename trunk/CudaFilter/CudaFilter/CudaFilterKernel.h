//------------------------------------------------------------------------------
// File: CudaFilterKernel.h
// 
// Author: Ren Yifei, Lin Ziya
//
// Contact: yfren@cs.hku.hk, zlin@cs.hku.hk
//
// Desc: The macro here decides which kind of filter should be compiled. 
// We could get different filter by define specific macro, one at a time.
// Some CUDA funtion declarations are also here.
//
//------------------------------------------------------------------------------

#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_

#include <cuda.h>
#include <cutil.h>

typedef unsigned char BYTE;

/* Decide which filter to compile */
//------------------------------------------------------------------------------

//#define COMPILE_SOBEL_FILTER
//#define COMPILE_LAPLACIAN_FILTER
#define COMPILE_AVERAGE_FILTER
//#define COMPILE_HIGH_BOOST_FILTER

//------------------------------------------------------------------------------

bool CUDAInit(int width, int height);

void CUDARelease();

bool CUDABeginDetection(BYTE* pImageIn, long dataLength);

bool CUDAEndDetection(BYTE* pImageOut);

#endif