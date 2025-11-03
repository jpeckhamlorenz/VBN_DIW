//Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.
/** @file
@brief	LJXA Profile Data Converter Header
*/

#ifndef _PROFILE_DATA_CONVERT_H_
#define _PROFILE_DATA_CONVERT_H_

#include <vector>
#include "LJX8_IF_Linux.h"

#define	PP_ERR_NONE					(0)		/* normal */
#define	PP_ERR_ARGUMENT				(1)		/* parameter error */

#define PROFILE_HEADER_SIZE_IN_INT	(6)
#define BRIGHTNESS_VALUE			(0x40)

/* definition related to raw level packet */
#define DEF_NOTIFY_MAGIC_NO		(0xFFFFFFFFU)
#define	DEF_PROF_ALARM_20		(524288)		/* 20bit : 0x00080000 */
#define	DEF_PROF_INVALID_20		(524289)		/* 20bit : 0x00080001 */
#define	DEF_PROF_BLIND_20		(524290)		/* 20bit : 0x00080002 */
#define	DEF_PROF_UNPROCESS_20	(524291)		/* 20bit : 0x00080003 */

#define	DEF_PROF_ALARM			(-2147483647-1)	/* 32bit : 0x80000000 */
#define	DEF_PROF_INVALID		(-2147483647)	/* 32bit : 0x80000001 */
#define	DEF_PROF_BLIND			(-2147483646)	/* 32bit : 0x80000002 */
#define	DEF_PROF_UNPROCESS		(-2147483645)	/* 32bit : 0x80000003 */
#define DEF_PROF_VALIDDATA_MIN	(-2147483644)	/* 32bit : 0x80000004 */

class ProfileDataConvert
{
public:
	static unsigned char ConvertProfileData20to32(const unsigned char profileType, const unsigned short wProfCnt, const unsigned short wUnit, const void* const profileData, int* convertedZProfile);
	
	static void ConvertProfileData32to16AsSimpleArray(int* pdwBatchData, unsigned short profileSize, int withLuminance, unsigned int getProfileCount, unsigned short zUnit, LJX8IF_PROFILE_HEADER* pProfileHeaderArray, unsigned short* pHeightProfileArray, unsigned short* pLuminanceProfileArray);

	static int ConvertProfileData(unsigned char dataCount, unsigned char byResProfType, int nProfileDataMargin, const unsigned short wProfCnt,const unsigned short wUnit,const void* pInData, unsigned int* pOutZProf, unsigned int dwDataSize);
	static unsigned char GetProfileCount(unsigned char byKind);
	
private:
	static void Convert20to32(const unsigned int dataSize, const unsigned int* readData, unsigned int* readPosition, unsigned int* brightnessOffset, unsigned int* convertedData);
};

#endif // _PROFILE_DATA_CONVERT_H_