//------------------------------------------------------------------------------
// File: FilterGUID.h
// 
// Author: Ren Yifei
// 
// Desc:
//
//
//------------------------------------------------------------------------------

#ifndef _FILTER_GUID_H_
#define _FILTER_GUID_H_

// {CADEEAFA-DBCA-4a84-BE62-4BB6E834579C}
DEFINE_GUID(CLSID_CudaSobelFilter, 
			0xcadeeafa, 0xdbca, 0x4a84, 0xbe, 0x62, 0x4b, 0xb6, 0xe8, 0x34, 0x57, 0x9c);

// {93C27220-40DD-473f-9831-6ABDF5BA3A2E}
DEFINE_GUID(CLSID_CudaLaplacianFilter, 
			0x93c27220, 0x40dd, 0x473f, 0x98, 0x31, 0x6a, 0xbd, 0xf5, 0xba, 0x3a, 0x2e);

// {43911DFD-AC00-481b-A4D6-80F19BE757AB}
DEFINE_GUID(CLSID_CudaAverageFilter, 
			0x43911dfd, 0xac00, 0x481b, 0xa4, 0xd6, 0x80, 0xf1, 0x9b, 0xe7, 0x57, 0xab);


// {DE09DC78-6CC6-4187-9F83-534D58FE26E0}
DEFINE_GUID(CLSID_CudaHighBoostFilter, 
			0xde09dc78, 0x6cc6, 0x4187, 0x9f, 0x83, 0x53, 0x4d, 0x58, 0xfe, 0x26, 0xe0);


#endif
