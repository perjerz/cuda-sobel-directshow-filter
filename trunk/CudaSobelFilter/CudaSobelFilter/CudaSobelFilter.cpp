//------------------------------------------------------------------------------
// File: CudaSobelFilter.cpp
// 
// Author: Ren Yifei
// 
// Desc: 
// 
//
//------------------------------------------------------------------------------

#include <streams.h>
#include <windows.h>
#include <initguid.h>
#include <olectl.h>
#include <cuda.h>

#if (1100 > _MSC_VER)
#include <olectlid.h>
#endif

#include "FilterGUID.h"
#include "CudaSobelFilter.h"

extern "C"
{
	#include "CudaSobelKernel.h"
}

#pragma warning(disable:4238)  // nonstd extension used: class rvalue used as lvalue

const AMOVIESETUP_MEDIATYPE sudPinTypes =
{
	&MEDIATYPE_Video,       // Major type
	&MEDIASUBTYPE_NULL      // Minor type
};

const AMOVIESETUP_PIN psudPins[] =
{
	{
		L"Input",           // String pin name
		FALSE,              // Is it rendered
		FALSE,              // Is it an output
		FALSE,              // Allowed none
		FALSE,              // Allowed many
		&CLSID_NULL,        // Connects to filter
		L"Output",          // Connects to pin
		1,                  // Number of types
		&sudPinTypes		// The pin details
	},

	{ 
		L"Output",          // String pin name
		FALSE,              // Is it rendered
		TRUE,               // Is it an output
		FALSE,              // Allowed none
		FALSE,              // Allowed many
		&CLSID_NULL,        // Connects to filter
		L"Input",           // Connects to pin
		1,                  // Number of types
		&sudPinTypes        // The pin details
	}
};

const AMOVIESETUP_FILTER sudContrast =
{
	&CLSID_CudaSobelFilter,        // Filter CLSID
	L"CudaSobelFilter",      // Filter name
	MERIT_DO_NOT_USE,       // Its merit
	2,                      // Number of pins
	psudPins                // Pin details
};

CFactoryTemplate g_Templates[1] = 
{
	{ 
		L"CudaSobelFilter"
		, &CLSID_CudaSobelFilter
		, CudaSobelFilter::CreateInstance
		, NULL
		, &sudContrast 
	}
};
int g_cTemplates = sizeof(g_Templates) / sizeof(g_Templates[0]);



//
// Constructor
//
CudaSobelFilter::CudaSobelFilter(TCHAR *tszName,LPUNKNOWN punk,HRESULT *phr) :
CTransformFilter(tszName, punk, CLSID_CudaSobelFilter)
{
	ASSERT(tszName);
	ASSERT(phr);

}

CUnknown * WINAPI CudaSobelFilter::CreateInstance(LPUNKNOWN punk, HRESULT *phr) 
{
	ASSERT(phr);

	CudaSobelFilter *pNewObject = new CudaSobelFilter(NAME("CudaSobelFilter"), punk, phr);
	if (pNewObject == NULL) 
	{
		if (phr)
			*phr = E_OUTOFMEMORY;
	}

	return pNewObject;

} // CreateInstance


//
// Transform
//
// Copy the input sample into the output sample
// Then transform the output sample 'in place'
//
HRESULT CudaSobelFilter::Transform(IMediaSample *pIn, IMediaSample *pOut)
{
	HRESULT hr = Copy(pIn, pOut);
	
	if (FAILED(hr))
		return hr;

	return Transform(pOut);

} // Transform


//
// Copy
//
// Make destination an identical copy of source
//
HRESULT CudaSobelFilter::Copy(IMediaSample *pSource, IMediaSample *pDest) const
{
	CheckPointer(pSource,E_POINTER);
	CheckPointer(pDest,E_POINTER);

	// Copy the sample data
	BYTE *pSourceBuffer, *pDestBuffer;
	long lSourceSize = pSource->GetActualDataLength();

#ifdef DEBUG
	long lDestSize = pDest->GetSize();
	ASSERT(lDestSize >= lSourceSize);
#endif

	pSource->GetPointer(&pSourceBuffer);
	pDest->GetPointer(&pDestBuffer);

	CopyMemory((PVOID) pDestBuffer,(PVOID) pSourceBuffer,lSourceSize);

	// Copy the sample times

	REFERENCE_TIME TimeStart, TimeEnd;
	if(NOERROR == pSource->GetTime(&TimeStart, &TimeEnd))
	{
		pDest->SetTime(&TimeStart, &TimeEnd);
	}

	LONGLONG MediaStart, MediaEnd;
	if(pSource->GetMediaTime(&MediaStart,&MediaEnd) == NOERROR)
	{
		pDest->SetMediaTime(&MediaStart,&MediaEnd);
	}

	// Copy the Sync point property

	HRESULT hr = pSource->IsSyncPoint();
	if(hr == S_OK)
	{
		pDest->SetSyncPoint(TRUE);
	}
	else if(hr == S_FALSE)
	{
		pDest->SetSyncPoint(FALSE);
	}
	else
	{  // an unexpected error has occured...
		return E_UNEXPECTED;
	}

	// Copy the media type

	AM_MEDIA_TYPE *pMediaType;
	pSource->GetMediaType(&pMediaType);
	pDest->SetMediaType(pMediaType);
	DeleteMediaType(pMediaType);

	// Copy the preroll property

	hr = pSource->IsPreroll();
	if(hr == S_OK)
	{
		pDest->SetPreroll(TRUE);
	}
	else if(hr == S_FALSE)
	{
		pDest->SetPreroll(FALSE);
	}
	else
	{  // an unexpected error has occured...
		return E_UNEXPECTED;
	}

	// Copy the discontinuity property

	hr = pSource->IsDiscontinuity();

	if(hr == S_OK)
	{
		pDest->SetDiscontinuity(TRUE);
	}
	else if(hr == S_FALSE)
	{
		pDest->SetDiscontinuity(FALSE);
	}
	else
	{  // an unexpected error has occured...
		return E_UNEXPECTED;
	}

	// Copy the actual data length

	long lDataLength = pSource->GetActualDataLength();
	pDest->SetActualDataLength(lDataLength);

	return NOERROR;

} // Copy


//
// Transform
//
// 'In place' adjust the contrast of this sample
//
HRESULT CudaSobelFilter::Transform(IMediaSample *pMediaSample)
{
	CheckPointer(pMediaSample,E_POINTER);

	AM_MEDIA_TYPE *pAdjustedType = NULL;

	pMediaSample->GetMediaType(&pAdjustedType);

	if(pAdjustedType != NULL)
	{
		if(CheckInputType(&CMediaType(*pAdjustedType)) == S_OK)
		{
			m_pInput->CurrentMediaType() = *pAdjustedType;
			CoTaskMemFree(pAdjustedType);
		}
		else
		{
			CoTaskMemFree(pAdjustedType);
			return E_FAIL;
		}
	}

	// Pass on format changes to downstream filters
	
	//testing 此处未获得需要的mediatype，可能是下游的错？？？

	//if(pAdjustedType != NULL)
	{
		CMediaType AdjustedType((AM_MEDIA_TYPE) m_pInput->CurrentMediaType());

		HRESULT hr = DetectSobelEdge(pMediaSample);
		
		if(hr == S_OK)
		{
			pMediaSample->SetMediaType(&AdjustedType);
		}
		else
		{
			return hr;
		}
	}

	return NOERROR;

} // Transform

HRESULT CudaSobelFilter::DetectSobelEdge( IMediaSample *pMediaSample )
{
	long dataLength = pMediaSample->GetActualDataLength();

	BYTE* pImageBuffer;

	pMediaSample->GetPointer(&pImageBuffer);
	
	if(!CUDABeginDetection(pImageBuffer, dataLength))
		return E_FAIL;

	if(!CUDAEndDetection(pImageBuffer))
		return E_FAIL;

	return S_OK;
}

//
// CheckInputType
//
// Check the input type is OK, return an error otherwise
//
HRESULT CudaSobelFilter::CheckInputType(const CMediaType *mtIn)
{
	CheckPointer(mtIn,E_POINTER);

	// Check this is a VIDEOINFO type

	if(*mtIn->FormatType() != FORMAT_VideoInfo)
	{
		return E_INVALIDARG;
	}

	if(	!IsEqualGUID(*mtIn->Type(), MEDIATYPE_Video) ||	!IsEqualGUID(*mtIn->Subtype(), MEDIASUBTYPE_IYUV))
	{
		return E_INVALIDARG;
	}

	return NOERROR;

} // CheckInputType


//
// CheckTransform
//
// To be able to transform the formats must be identical
//
HRESULT CudaSobelFilter::CheckTransform(const CMediaType *mtIn,const CMediaType *mtOut)
{
	CheckPointer(mtIn,E_POINTER);
	CheckPointer(mtOut,E_POINTER);

	HRESULT hr;
	if(FAILED(hr = CheckInputType(mtIn)))
	{
		return hr;
	}

	// format must be a VIDEOINFOHEADER
	if(*mtOut->FormatType() != FORMAT_VideoInfo)
	{
		return E_INVALIDARG;
	}

	// formats must be big enough 
	if(	mtIn->FormatLength() < sizeof(VIDEOINFOHEADER) ||
		mtOut->FormatLength() < sizeof(VIDEOINFOHEADER))
		return E_INVALIDARG;

	VIDEOINFO *pInput  = (VIDEOINFO *) mtIn->Format();
	VIDEOINFO *pOutput = (VIDEOINFO *) mtOut->Format();

	m_ImageWidth = pInput->bmiHeader.biWidth;
	m_ImageHeight = pInput->bmiHeader.biHeight;
	
	//testing!! release ??? 在最后释放
	if(!CUDAInit(m_ImageWidth, m_ImageHeight))
		return E_INVALIDARG;

	if(memcmp(&pInput->bmiHeader,&pOutput->bmiHeader,sizeof(BITMAPINFOHEADER)) == 0)
	{
		return NOERROR;
	}

	return E_INVALIDARG;

} // CheckTransform


//
// DecideBufferSize
//
// Tell the output pin's allocator what size buffers we
// require. Can only do this when the input is connected
//
HRESULT CudaSobelFilter::DecideBufferSize(IMemAllocator *pAlloc,ALLOCATOR_PROPERTIES *pProperties)
{
	CheckPointer(pAlloc,E_POINTER);
	CheckPointer(pProperties,E_POINTER);

	// Is the input pin connected

	if(m_pInput->IsConnected() == FALSE)
	{
		return E_UNEXPECTED;
	}

	HRESULT hr = NOERROR;
	pProperties->cBuffers = 1;
	pProperties->cbBuffer = m_pInput->CurrentMediaType().GetSampleSize();

	ASSERT(pProperties->cbBuffer);

	//testing!
	pProperties->cbBuffer = 10000000;

	// If we don't have fixed sized samples we must guess some size

	if(!m_pInput->CurrentMediaType().bFixedSizeSamples)
	{
		if(pProperties->cbBuffer < 100000)
		{
			// nothing more than a guess!!
			pProperties->cbBuffer = 100000;
		}
	}

	// Ask the allocator to reserve us some sample memory, NOTE the function
	// can succeed (that is return NOERROR) but still not have allocated the
	// memory that we requested, so we must check we got whatever we wanted

	ALLOCATOR_PROPERTIES Actual;

	hr = pAlloc->SetProperties(pProperties,&Actual);
	if(FAILED(hr))
	{
		return hr;
	}

	ASSERT(Actual.cBuffers == 1);

	if(pProperties->cBuffers > Actual.cBuffers ||
		pProperties->cbBuffer > Actual.cbBuffer)
	{
		return E_FAIL;
	}

	return NOERROR;

} // DecideBufferSize


//
// GetMediaType
//
// I support one type, namely the type of the input pin
// We must be connected to support the single output type
//
HRESULT CudaSobelFilter::GetMediaType(int iPosition, CMediaType *pMediaType)
{
	// Is the input pin connected

	if(m_pInput->IsConnected() == FALSE)
	{
		return E_UNEXPECTED;
	}

	// This should never happen

	if(iPosition < 0)
	{
		return E_INVALIDARG;
	}

	// Do we have more items to offer

	if(iPosition > 0)
	{
		return VFW_S_NO_MORE_ITEMS;
	}

	CheckPointer(pMediaType,E_POINTER);

	*pMediaType = m_pInput->CurrentMediaType();
	return NOERROR;

} // GetMediaType

//
// DllRegisterServer
//
// Handle registration of this filter
//
STDAPI DllRegisterServer()
{
	return AMovieDllRegisterServer2(TRUE);

} // DllRegisterServer


//
// DllUnregisterServer
//
STDAPI DllUnregisterServer()
{
	return AMovieDllRegisterServer2(FALSE);

} // DllUnregisterServer


//
// DllEntryPoint
//
extern "C" BOOL WINAPI DllEntryPoint(HINSTANCE, ULONG, LPVOID);

BOOL APIENTRY DllMain(HANDLE hModule, 
					  DWORD  dwReason, 
					  LPVOID lpReserved)
{
	return DllEntryPoint((HINSTANCE)(hModule), dwReason, lpReserved);
}