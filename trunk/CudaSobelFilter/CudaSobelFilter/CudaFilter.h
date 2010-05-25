//------------------------------------------------------------------------------
// File: CudaFilter.h
// 
// Author: Ren Yifei, Lin Ziya
//
// Contact: yfren@cs.hku.hk, zlin@cs.hku.hk
//
// Desc: CUDA Filter class derived from CTransformFilter in DirectShow. 
// The specific type is decided by the macro in CudAFilterKernel.h
//
//------------------------------------------------------------------------------

#ifndef _CUDA_FILTER_H_
#define _CUDA_FILTER_H_

class CudaTransformFilter : public CTransformFilter
{
public:

	static CUnknown * WINAPI CreateInstance(LPUNKNOWN punk, HRESULT *phr);

	HRESULT Transform(IMediaSample *pIn, IMediaSample *pOut);
	HRESULT CheckInputType(const CMediaType *mtIn);
	HRESULT CheckTransform(const CMediaType *mtIn,const CMediaType *mtOut);
	HRESULT GetMediaType(int iPosition, CMediaType *pMediaType);
	HRESULT DecideBufferSize(IMemAllocator *pAlloc, ALLOCATOR_PROPERTIES *pProperties);

private:

	CudaTransformFilter(TCHAR *tszName, LPUNKNOWN punk, HRESULT *phr);

	HRESULT Copy(IMediaSample *pSource, IMediaSample *pDest) const;
	HRESULT Transform(IMediaSample *pMediaSample);

	HRESULT ApplyFilter(IMediaSample *pMediaSample);

	unsigned int m_ImageWidth;
	unsigned int m_ImageHeight;
};

#endif