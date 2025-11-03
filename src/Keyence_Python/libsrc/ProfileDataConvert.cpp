//Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.
/** @file
@brief	LJXA Profile Data Converter Implementation
*/

#include <limits.h>
#include <string.h>
#include <math.h>
#include "ProfileDataConvert.h"
#include "LJX8_ErrorCode.h"

#define sizeofBytes(size,bytes)	(((size) + ((bytes * 8) - 1U)) / (bytes * 8))

const static unsigned char MASK_PROFILE_COUNT = 0x60;
const static double PROFILE_CONVERT_UNIT = 8.0;

const static int BIT_PER_BYTE = 8;
const static int PROFILE_BIT_SIZE = 20;
const static int BRIGHTNESS_BIT_SIZE = 10;
const static int PACKING_PROFILE_DATA_SIZE_PER_BYTE = 4;

unsigned char ProfileDataConvert::ConvertProfileData20to32(
	const unsigned char profileType, 
	const unsigned short wProfCnt, 
	const unsigned short wUnit, 
	const void* const profileData, 
	int* convertedZProfile)
{
	unsigned int* readProfileData = NULL;
	unsigned int* convertedProfileData = NULL;
	unsigned short profileSize = 0U;
	unsigned int readPosition = 0U;
	unsigned short writePosition = 0U;
	int i = 0;

	if ((NULL == profileData) || (NULL == convertedZProfile))
	{
		return PP_ERR_ARGUMENT;
	}

	convertedProfileData = (unsigned int*)convertedZProfile;
	readProfileData = (unsigned int*)profileData;

	//Offset pointer to the location where the profile data is stored
	readProfileData += PROFILE_HEADER_SIZE_IN_INT;

	profileSize = wProfCnt;
	int profileUnit = wUnit;
	
	unsigned int brightnessOffset = 0U;

	//Convert 20-bit data to 32-bit data
	Convert20to32(profileSize, readProfileData, &readPosition, &brightnessOffset, convertedProfileData);

	for (i = 0; i< profileSize; i++)
	{
		//Convert irregular value
		if (DEF_PROF_ALARM_20 == convertedProfileData[i]) {
			convertedProfileData[i] = DEF_PROF_ALARM;
		}
		else if (DEF_PROF_INVALID_20 == convertedProfileData[i]) {
			convertedProfileData[i] = DEF_PROF_INVALID;
		}
		else if (DEF_PROF_BLIND_20 == convertedProfileData[i]) {
			convertedProfileData[i] = DEF_PROF_BLIND;
		}
		else if (DEF_PROF_UNPROCESS_20 == convertedProfileData[i]) {
			convertedProfileData[i] = DEF_PROF_UNPROCESS;
		}
		else {
			//Extend the sign
			if (0x00080000 & convertedProfileData[i]) {
				convertedProfileData[i] |= 0xFFF00000;
			}
			//Convert to a value in units of 0.01um
			convertedProfileData[i] *= profileUnit;
		}
	}

	int isBrightness = (profileType & BRIGHTNESS_VALUE) == BRIGHTNESS_VALUE;
	if (isBrightness)
	{
		//Offset pointer to the location where the luminance profile data is stored
		readProfileData += brightnessOffset;
		convertedProfileData += profileSize;

		//If the number of profiles is not divisible by 8 (e.g. 300), the remainder is converted first.
		int isProfileCount300 = profileSize % 8 != 0;
		if (isProfileCount300) {
			readPosition = 0U;
			writePosition = 0U;
			
			//To process the remaining luminance profile
			readProfileData -= 1;

			convertedProfileData[writePosition + 0] = (0xFC000000 & readProfileData[readPosition + 0U]) >> 26 | (0x0000000F & readProfileData[readPosition + 1U]) << 6;
			convertedProfileData[writePosition + 1] = (0x03FF0000 & readProfileData[readPosition + 0U]) >> 16;
			convertedProfileData[writePosition + 2] = (0x00FFC000 & readProfileData[readPosition + 1U]) >> 14;
			convertedProfileData[writePosition + 3] = (0x00003FF0 & readProfileData[readPosition + 1U]) >> 4;
			convertedProfileData[writePosition + 4] = (0x00000FFC & readProfileData[readPosition + 2U]) >> 2;
			convertedProfileData[writePosition + 5] = (0xFF000000 & readProfileData[readPosition + 1U]) >> 24 | (0x00000003 & readProfileData[readPosition + 2U]) << 8;
			convertedProfileData[writePosition + 6] = (0xFFC00000 & readProfileData[readPosition + 2U]) >> 22;
			convertedProfileData[writePosition + 7] = (0x003FF000 & readProfileData[readPosition + 2U]) >> 12;
			
			readProfileData += 3;

			const int halfProfileData = 16 / 2;
			convertedProfileData += halfProfileData;
			profileSize -= halfProfileData;
		}

		for (readPosition = 0U, writePosition = 0U; writePosition < profileSize; readPosition += 5U, writePosition += 16U)
		{
			convertedProfileData[writePosition + 0] = (0x000FFC00 & readProfileData[readPosition + 0U]) >> 10;
			if (profileSize <= writePosition + 1) {
				break;
			}
			convertedProfileData[writePosition + 1] = (0x000003FF & readProfileData[readPosition + 0U]);
			if (profileSize <= writePosition + 2) {
				break;
			}
			convertedProfileData[writePosition + 2] = (0xC0000000 & readProfileData[readPosition + 0U]) >> 30 | (0x000000FF & readProfileData[readPosition + 1U]) << 2;
			if (profileSize <= writePosition + 3) {
				break;
			}
			convertedProfileData[writePosition + 3] = (0x3FF00000 & readProfileData[readPosition + 0U]) >> 20;
			if (profileSize <= writePosition + 4) {
				break;
			}
			convertedProfileData[writePosition + 4] = (0x0FFC0000 & readProfileData[readPosition + 1U]) >> 18;
			if (profileSize <= writePosition + 5) {
				break;
			}
			convertedProfileData[writePosition + 5] = (0x0003FF00 & readProfileData[readPosition + 1U]) >> 8;
			if (profileSize <= writePosition + 6) {
				break;
			}
			convertedProfileData[writePosition + 6] = (0x0000FFC0 & readProfileData[readPosition + 2U]) >> 6;
			if (profileSize <= writePosition + 7) {
				break;
			}
			convertedProfileData[writePosition + 7] = (0xF0000000 & readProfileData[readPosition + 1U]) >> 28 | (0x0000003F & readProfileData[readPosition + 2U]) << 4;
			if (profileSize <= writePosition + 8) {
				break;
			}
			convertedProfileData[writePosition + 8] = (0xFC000000 & readProfileData[readPosition + 2U]) >> 26 | (0x0000000F & readProfileData[readPosition + 3U]) << 6;
			if (profileSize <= writePosition + 9) {
				break;
			}
			convertedProfileData[writePosition + 9] = (0x03FF0000 & readProfileData[readPosition + 2U]) >> 16;
			if (profileSize <= writePosition + 10) {
				break;
			}
			convertedProfileData[writePosition + 10] = (0x00FFC000 & readProfileData[readPosition + 3U]) >> 14;
			if (profileSize <= writePosition + 11) {
				break;
			}
			convertedProfileData[writePosition + 11] = (0x00003FF0 & readProfileData[readPosition + 3U]) >> 4;
			if (profileSize <= writePosition + 12) {
				break;
			}
			convertedProfileData[writePosition + 12] = (0x00000FFC & readProfileData[readPosition + 4U]) >> 2;
			if (profileSize <= writePosition + 13) {
				break;
			}
			convertedProfileData[writePosition + 13] = (0xFF000000 & readProfileData[readPosition + 3U]) >> 24 | (0x00000003 & readProfileData[readPosition + 4U]) << 8;
			if (profileSize <= writePosition + 14) {
				break;
			}
			convertedProfileData[writePosition + 14] = (0xFFC00000 & readProfileData[readPosition + 4U]) >> 22;
			if (profileSize <= writePosition + 15) {
				break;
			}
			convertedProfileData[writePosition + 15] = (0x003FF000 & readProfileData[readPosition + 4U]) >> 12;
			if (profileSize <= writePosition + 16) {
				break;
			}
		}
	}
	return PP_ERR_NONE;
}

void ProfileDataConvert::Convert20to32(const unsigned int dataSize, const unsigned int* readData, unsigned int* readPosition, unsigned int* brightnessOffset, unsigned int* convertedData)
{
	unsigned int writePosition = 0U;

	for (*readPosition = 0U, writePosition = 0U; writePosition < dataSize; *readPosition += 5U, writePosition += 8U)
	{
		convertedData[writePosition + 0] = (0x000FFFFF & readData[*readPosition + 0U]);
		if (dataSize <= writePosition + 1) {
			*brightnessOffset = *readPosition + 1U;
			break;
		}
		convertedData[writePosition + 1] = ((0xFFF00000 & readData[*readPosition + 0U]) >> 20) | ((0x000000FF & readData[*readPosition + 1U]) << 12);
		if (dataSize <= writePosition + 2) {
			*brightnessOffset = *readPosition + 2U;
			break;
		}
		convertedData[writePosition + 2] = (0x0FFFFF00 & readData[*readPosition + 1U]) >> 8;
		if (dataSize <= writePosition + 3) {
			*brightnessOffset = *readPosition + 2U;
			break;
		}
		convertedData[writePosition + 3] = ((0xF0000000 & readData[*readPosition + 1U]) >> 28) | ((0x0000FFFF & readData[*readPosition + 2U]) << 4);
		if (dataSize <= writePosition + 4) {
			*brightnessOffset = *readPosition + 3U;
			break;
		}
		convertedData[writePosition + 4] = ((0xFFFF0000 & readData[*readPosition + 2U]) >> 16) | ((0x0000000F & readData[*readPosition + 3U]) << 16);
		if (dataSize <= writePosition + 5) {
			*brightnessOffset = *readPosition + 4U;
			break;
		}
		convertedData[writePosition + 5] = (0x00FFFFF0 & readData[*readPosition + 3U]) >> 4;
		if (dataSize <= writePosition + 6) {
			*brightnessOffset = *readPosition + 4U;
			break;
		}
		convertedData[writePosition + 6] = ((0xFF000000 & readData[*readPosition + 3U]) >> 24) | ((0x00000FFF & readData[*readPosition + 4U]) << 8);
		if (dataSize <= writePosition + 7) {
			*brightnessOffset = *readPosition + 5U;
			break;
		}
		convertedData[writePosition + 7] = (0xFFFFF000 & readData[*readPosition + 4U]) >> 12;
		if (dataSize <= writePosition + 8) {
			*brightnessOffset = *readPosition + 5U;
			break;
		}
	}
}

void ProfileDataConvert::ConvertProfileData32to16AsSimpleArray(
	int* pdwBatchData, 
	unsigned short profileSize, 
	int withLuminance, 
	unsigned int getProfileCount, 
	unsigned short zUnit, 
	LJX8IF_PROFILE_HEADER* pProfileHeaderArray, 
	unsigned short* pHeightProfileArray, 
	unsigned short* pLuminanceProfileArray)
{
	const int coefficient = zUnit * 8;

	const auto headerSize = sizeof(LJX8IF_PROFILE_HEADER) / sizeof(int);
	const auto footerSize = sizeof(LJX8IF_PROFILE_FOOTER) / sizeof(int);

	const auto headerByteSize = sizeof(LJX8IF_PROFILE_HEADER);
	//const auto profileByteSize = profileSize * sizeof(int);

	auto source = pdwBatchData;
	for (unsigned int i = 0; i < getProfileCount; i++) {
		const auto profileIndexBase = profileSize * i;

		// Header
		memcpy(pProfileHeaderArray + i, source, headerByteSize);
		source += headerSize;

		// Height data
		for (auto k = 0; k < profileSize; k++) {
			auto data = (long)source[k];
			if (data < DEF_PROF_VALIDDATA_MIN) {
				// Irregular data is converted to zero.
				pHeightProfileArray[profileIndexBase + k] = 0;
				continue;
			}

			auto value = data / coefficient;
			if (value < SHRT_MIN || SHRT_MAX < value) {
				// Height data values from long-distance heads can exceed the 16-bit range,
				pHeightProfileArray[profileIndexBase + k] = 0;
				continue;
			}
			
			// To convert signed 16-bit data to unsigned data, offset the data by 32768
			pHeightProfileArray[profileIndexBase + k] = (short)(value + SHRT_MAX + 1);
		}
		source += profileSize;

		// Luminance data
		if (withLuminance) {
			for (auto k = 0; k < profileSize; k++) {
				pLuminanceProfileArray[profileIndexBase + k] = (short)source[k];
			}

			source += profileSize;
		}

		// Footer. 
		source += footerSize;
	}
}

int ProfileDataConvert::ConvertProfileData(
	unsigned char dataCount, 
	unsigned char byResProfType, 
	int nProfileDataMargin, 
	const unsigned short wProfCnt,
	const unsigned short wUnit,
	const void* pInData, 
	unsigned int* pOutZProf, 
	unsigned int dwDataSize)
{
	// Calculate profile size
	int isBrightness = (BRIGHTNESS_VALUE & byResProfType)>0 ? 1 : 0;
	int nMultipleValue = (isBrightness) ? 2 : 1;
	int nHeadCount = ProfileDataConvert::GetProfileCount(byResProfType);
	
	int nProfileUnitSize = wProfCnt * nMultipleValue * sizeof(unsigned int);
	int nProfileDataSize = nProfileUnitSize * nHeadCount;
	
	// Since conversion is performed in units of 8, the data size is prepared by rounding up to a multiple of 8.
	int nProfileBufferSize = (int)(ceil(wProfCnt / PROFILE_CONVERT_UNIT) * PROFILE_CONVERT_UNIT * nMultipleValue * sizeof(unsigned int) * nHeadCount);

	int nHeaderSize = sizeof(LJX8IF_PROFILE_HEADER);
	int nFooterSize = sizeof(LJX8IF_PROFILE_FOOTER);


	// Check if the user buffer size is enough
	if (dwDataSize < (unsigned int)((nProfileDataSize + nHeaderSize + nFooterSize) * dataCount)) return LJX8IF_RC_ERR_BUFFER_SHORT;
	
	const unsigned char* pProfileRawData = (const unsigned char*)pInData;
	unsigned char* pOut = (unsigned char*)pOutZProf;

	int nProfileDataRawSize = isBrightness ?
		sizeofBytes((wProfCnt * PROFILE_BIT_SIZE) + (wProfCnt * BRIGHTNESS_BIT_SIZE), PACKING_PROFILE_DATA_SIZE_PER_BYTE) * nHeadCount * PACKING_PROFILE_DATA_SIZE_PER_BYTE :
		sizeofBytes((wProfCnt * PROFILE_BIT_SIZE), PACKING_PROFILE_DATA_SIZE_PER_BYTE) * nHeadCount * PACKING_PROFILE_DATA_SIZE_PER_BYTE;

	for (int i = 0; i < dataCount; i++)
	{
		// temporary buffer for one profile
		unsigned char* aProfileData = new(std::nothrow) unsigned char[nProfileBufferSize];
		if (aProfileData == NULL) return LJX8IF_RC_ERR_NOMEMORY;
		memset(aProfileData, 0, nProfileDataSize);

		// 20-bit to 32-bit conversion
		if(ProfileDataConvert::ConvertProfileData20to32(byResProfType, wProfCnt, wUnit, pProfileRawData, (int*)aProfileData) != PP_ERR_NONE)
		{
			delete [] aProfileData;
			aProfileData = NULL;
			return LJX8IF_RC_ERR_PARAMETER;
		}

		// header
		memcpy(pOut, pProfileRawData, nHeaderSize);
		pOut += nHeaderSize;

		// profile data
		memcpy(pOut, aProfileData, nProfileDataSize);
		pOut += nProfileDataSize;

		// footer
		memcpy(pOut, (pProfileRawData + nHeaderSize + nProfileDataRawSize), nFooterSize);
		pOut += nFooterSize;

		pProfileRawData += nHeaderSize + nProfileDataRawSize + nFooterSize + nProfileDataMargin;
		delete[] aProfileData;
		aProfileData = NULL;
	}

	return LJX8IF_RC_OK;
}


unsigned char ProfileDataConvert::GetProfileCount(unsigned char byKind){
	unsigned char byTemp = byKind & (~MASK_PROFILE_COUNT);
	unsigned char byProfileCnt = 0;
	for (int i = 0; i < BIT_PER_BYTE; i++)
	{
		if (byTemp & (0x01 << i)) byProfileCnt++;
	}
	return byProfileCnt;
}