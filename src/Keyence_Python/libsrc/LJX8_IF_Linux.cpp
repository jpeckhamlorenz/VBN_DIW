//Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.
/** @file
@brief	LJX8_IF_Linux Implementation
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>

#include "LJX8_IF_Linux.h"
#include "LJX8_ErrorCode.h"
#include "ProfileDataConvert.h"

#define LJXA_CONNECT_TIMEOUT_SEC	(3U)
#define WRITE_DATA_SIZE				(20U * 1024U)
#define SEND_BUF_SIZE				(WRITE_DATA_SIZE + 16U)
#define RECEIVE_BUF_SIZE 			(3000U * 1024U)
#define FAST_RECEIVE_BUF_SIZE		(64U * 1024U)
#define	PARITY_ERROR				(0x40000000)

// Profile length
#define PROF_HEADER_LEN				(6U * sizeof(unsigned int))
#define PROF_CHECKSUM_LEN			(sizeof(unsigned int))
#define PROF_NOT_DATA_LEN			(PROF_HEADER_LEN + PROF_CHECKSUM_LEN)

#define RES_ADRESS_OFFSET_ERROR		(17U)
#define	PREFIX_ERROR_CONTROLLER		(0x8000)

#define divRoundUp(x, y)	(((x) + ((y) - 1U)) / (y))

const static unsigned int MASK_CURRENT_BATCH_COMMITED = 0x80000000;

/* NOTIFY packet */
typedef struct {
	unsigned int	dwMagicNo;	/* constant 0xFFFFFFFF */
	unsigned int	dwNotify;	/* NOTIFY */
} ST_CIND_PROFILE_NOTIFY;

typedef struct {
	unsigned char*	ReceiveBuffer;		// Buffer for passing data to the user finally in the callback function.
	unsigned char*	pbyTempBuffer;		// Buffer to store newly received raw data buffer.
	unsigned char*	pbyConvBuffer;		// Buffer to store 20bit profile (before conversion).
	unsigned char*	pbyConvEndBuffer;	// Buffer to store 32bit profile (after conversion).
	unsigned int	dwConvBufferIndex;	// How many data remains in the pbyConvBuffer, in bytes.
	int				bIsEnabled;			// Indicate that the thread is active.
	void			(*pFunc)(unsigned char*, unsigned int, unsigned int, unsigned int, unsigned int);
										// Callback function pointer.
	unsigned int	dwProfileMax;		// When this number of profiles acquired, it wake up the callback function.
	unsigned int	dwProfileLoopCnt;	// Number of profiles collected so far.
	unsigned int	dwProfileCnt;		// Number of data points for each profile.
	unsigned int	dwProfileSize;		// Size of each profile, in bytes. (20bit profile)
	unsigned int	dwProfileSize32;	// Size of each profile, in bytes. (32bit profile)
	unsigned short	wUnit;				// Unit of profile (in 0.01um)
	unsigned char	byKind;				// Kind of profile (e.g. luminance output is enabled)
	unsigned int	dwThreadId;			// Thread identifier
	//extended
	unsigned int	dwDeviceId;			// Device identifier (e.g. which number of head)
	int				nBufferSize;		// Size of ReceiveBuffer, in bytes.
	float			SimpleArrayZUnit_um;// Z unit of 16bit height image(in um)
	int				stopFlag;			// (Internal use) flag to terminate thread. 
	void			(*pFuncSimpleArray)(LJX8IF_PROFILE_HEADER* , unsigned short* , unsigned short* , unsigned int , unsigned int , unsigned int, unsigned int, unsigned int);
										// Callback function pointer(for Simple Array method)
} THREADPARAM_FAST;

// Static variable
static int sockfd[ MAX_LJXA_DEVICENUM ];
static int sockfdHighSpeed[ MAX_LJXA_DEVICENUM ];
static unsigned char mysendBuf[ MAX_LJXA_DEVICENUM ][ SEND_BUF_SIZE ];
static unsigned char myrecvline[ MAX_LJXA_DEVICENUM ][ RECEIVE_BUF_SIZE ];
static int  startcode[ MAX_LJXA_DEVICENUM ];
static pthread_t pthread[ MAX_LJXA_DEVICENUM ];
THREADPARAM_FAST m_ThreadParamFast[ MAX_LJXA_DEVICENUM ];

// Function prototype
static int myAnyCommand(int lDeviceId, unsigned char * senddata, int dataLength);
static int sendSingleCommand(int lDeviceId, unsigned char CommandCode);
static void *receiveThread(void *args);
static void InitThreadParamFast(int lDeviceId);
static void SetThreadInitParamFast(int lDeviceId, void (*pCallBack)(unsigned char*, unsigned int, unsigned int, unsigned int, unsigned int), unsigned int dwCnt, unsigned int dwThreadId);
static int SetThreadParamFast(int lDeviceId, unsigned int dwProfileCnt, unsigned char byKind, unsigned short wUnit);

static unsigned int ReceiveDataCheck(unsigned char* pbyConvBeforeData, unsigned int dwRecvSize, unsigned int dwProfSize, unsigned char byKind, unsigned int dwProfCnt, unsigned short wUnit, unsigned char* pbyConvAfterData, unsigned int* pdwConvSize, unsigned int* pdwNotify, unsigned int dwNumReceived, unsigned char* pbyTempBuffer);

static void CallbackFuncWrapper( unsigned char *Buf, unsigned int dwSize ,unsigned int dwCount, unsigned int dwNotify, unsigned int  dwUser);

extern "C"
{
//-----------------------------------------------------------------
// LJX8_IF funcitons implementation
//-----------------------------------------------------------------
int LJX8IF_EthernetOpen(int lDeviceId, LJX8IF_ETHERNET_CONFIG* pEthernetConfig)
{
	char ip[15];
	unsigned char *ipn;
	struct sockaddr_in servaddr;
	
	if(pEthernetConfig == NULL)
		return LJX8IF_RC_ERR_PARAMETER;
	
	ipn= pEthernetConfig->abyIpAddress;
	
	sprintf(ip,"%d.%d.%d.%d",ipn[0],ipn[1],ipn[2],ipn[3]);
	
	// Create socket
	sockfd[lDeviceId] = socket(AF_INET,SOCK_STREAM,0);
	if( sockfd[lDeviceId] <0){
		return LJX8IF_RC_ERR_OPEN;
	}
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr=inet_addr(ip);
	servaddr.sin_port=htons(pEthernetConfig->wPortNo);

	// Connect socket in non-blocking mode to determine timeout
	// set non-blocking mode
	int flag;
	if( (flag = fcntl(sockfd[lDeviceId], F_GETFL, NULL)) <0){
		return LJX8IF_RC_ERR_OPEN;
	}
	if( fcntl(sockfd[lDeviceId], F_SETFL, flag | O_NONBLOCK) <0){
		return LJX8IF_RC_ERR_OPEN;
	}
	
	if( connect(sockfd[lDeviceId], (struct sockaddr *)&servaddr, sizeof(servaddr)) <0){
		if(errno  != EINPROGRESS){
			return LJX8IF_RC_ERR_OPEN;
		}
		
		struct timeval tv;
		int res;
		fd_set fds;

		tv.tv_sec = LJXA_CONNECT_TIMEOUT_SEC;
		tv.tv_usec = 0;
		FD_ZERO(&fds);
		FD_SET(sockfd[lDeviceId], &fds);

		res = select(sockfd[lDeviceId]+1, NULL, &fds, NULL, &tv);

		if(res <= 0){ // Connection timeout
			return LJX8IF_RC_ERR_OPEN;
		}
		else if(res >0){
			socklen_t optlen=sizeof(servaddr);
			
			// Check if socket is writable
			if(getpeername(sockfd[lDeviceId], (struct sockaddr *)&servaddr, &optlen) <0){
				return LJX8IF_RC_ERR_OPEN;
			}
		}
	}
	
	// Blocking mode
	if( fcntl(sockfd[lDeviceId], F_SETFL, flag) <0){
		return LJX8IF_RC_ERR_OPEN;
	}
	
  	return LJX8IF_RC_OK;
}

int LJX8IF_CommunicationClose(int lDeviceId)
{
	close(sockfdHighSpeed[lDeviceId]);	
	close(sockfd[lDeviceId]);
	return LJX8IF_RC_OK;
}

int LJX8IF_RebootController(int lDeviceId)
{
	return sendSingleCommand(lDeviceId, 0x02);
}

int LJX8IF_ReturnToFactorySetting(int lDeviceId)
{
	return sendSingleCommand(lDeviceId, 0x03);
}

int LJX8IF_ControlLaser(int lDeviceId, unsigned char byState)
{
	unsigned char senddata[] = { 0x2A, 0x00, 0x00, 0x00, byState , 0x00, 0x00, 0x00 };
	return myAnyCommand(lDeviceId, senddata, sizeof(senddata));
}

int LJX8IF_GetError(int lDeviceId, unsigned char byReceivedMax, unsigned char* pbyErrCount, unsigned short* pwErrCode)
{
	int res = sendSingleCommand(lDeviceId, 0x04);
	if(LJX8IF_RC_OK != res) return res;
	
	unsigned char numberOfError = myrecvline[lDeviceId][28];
	unsigned char copyNum;
	
	// Number of errors
	*pbyErrCount = numberOfError;
	
	// Copy error infomation
	if(numberOfError >= 1){
		if(byReceivedMax > numberOfError){
			copyNum = numberOfError;
		}
		else{
			copyNum = byReceivedMax;
		}
		memcpy(pwErrCode, &myrecvline[lDeviceId][32], sizeof(unsigned short) * copyNum );
	}

	return LJX8IF_RC_OK;
}

int LJX8IF_ClearError(int lDeviceId, unsigned short wErrCode)
{
	unsigned char *abyDat = (unsigned char*)&wErrCode;
	unsigned char senddata[] = { 0x05, 0x00, 0x00, 0x00, abyDat[0], abyDat[1], 0x00, 0x00 };
	return myAnyCommand(lDeviceId, senddata, sizeof(senddata));
}

int LJX8IF_TrgErrorReset(int lDeviceId)
{
	return sendSingleCommand(lDeviceId, 0x2B);
}

int LJX8IF_GetTriggerAndPulseCount(int lDeviceId, unsigned int* pdwTriggerCount, int* plEncoderCount)
{
	int res = sendSingleCommand(lDeviceId, 0x4B);
	if(LJX8IF_RC_OK != res) return res;
	
	*pdwTriggerCount 	= *(unsigned int*)(&myrecvline[lDeviceId][28]);
	*plEncoderCount 	= *(int*)(&myrecvline[lDeviceId][32]);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_GetHeadTemperature(int lDeviceId, short* pnSensorTemperature, short* pnProcessorTemperature, short* pnCaseTemperature)
{
	int res = sendSingleCommand(lDeviceId, 0x4C);
	if(LJX8IF_RC_OK != res) return res;

	*pnSensorTemperature 	= *(short*)(&myrecvline[lDeviceId][28]);
	*pnProcessorTemperature = *(short*)(&myrecvline[lDeviceId][30]);
	*pnCaseTemperature		= *(short*)(&myrecvline[lDeviceId][32]);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_GetHeadModel(int lDeviceId, char* pHeadModel)
{
	int res = sendSingleCommand(lDeviceId, 0x06);
	if(LJX8IF_RC_OK != res) return res;
	
	unsigned char numberOfUnit = myrecvline[lDeviceId][28];
	
	if(numberOfUnit >= 2){
		memcpy(pHeadModel, &myrecvline[lDeviceId][44 + 52], 32);
	}else{
		bzero(pHeadModel, 32);
	}
	
	return LJX8IF_RC_OK;
}

int LJX8IF_GetSerialNumber(int lDeviceId, char* pControllerSerialNo, char* pHeadSerialNo)
{
	int res = sendSingleCommand(lDeviceId, 0x06);
	if(LJX8IF_RC_OK != res) return res;
	
	unsigned char numberOfUnit = myrecvline[lDeviceId][28];
	
	if(numberOfUnit >= 1){
		memcpy(pControllerSerialNo, &myrecvline[lDeviceId][44 + 32], 16);
		
		if(numberOfUnit >= 2){
			memcpy(pHeadSerialNo, &myrecvline[lDeviceId][44 + 32 + 52], 16);
		}
	}else{
		bzero(pControllerSerialNo, 16);
		bzero(pHeadSerialNo, 16);
	}
	
	return LJX8IF_RC_OK;
}

int LJX8IF_GetAttentionStatus(int lDeviceId, unsigned short* pwAttentionStatus)
{
	int res = sendSingleCommand(lDeviceId, 0x65);
	if(LJX8IF_RC_OK != res) return res;
	
	*pwAttentionStatus 	= *(unsigned short*)(&myrecvline[lDeviceId][28]);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_Trigger(int lDeviceId)
{
	return sendSingleCommand(lDeviceId, 0x21);
}

int LJX8IF_StartMeasure(int lDeviceId)
{
	return sendSingleCommand(lDeviceId, 0x022);
}

int LJX8IF_StopMeasure(int lDeviceId)
{
	return sendSingleCommand(lDeviceId, 0x023);
}

int LJX8IF_ClearMemory(int lDeviceId)
{
	return sendSingleCommand(lDeviceId, 0x027);
}

int LJX8IF_SetSetting(int lDeviceId, unsigned char byDepth, LJX8IF_TARGET_SETTING TargetSetting, void* pData, unsigned int dwDataSize, unsigned int* pdwError)
{
	LJX8IF_TARGET_SETTING *tset = &TargetSetting;
	
	unsigned char header_dat[] = { 0x32, 0x00, 0x00, 0x00, 
								byDepth, 0x00, 0x00, 0x00, 
								tset->byType,
								tset->byCategory,
								tset->byItem,
								0x00,
								tset->byTarget1,
								tset->byTarget2,
								tset->byTarget3,
								tset->byTarget4};
								
	struct
	{
		unsigned char header[sizeof(header_dat)];
		unsigned char body[WRITE_DATA_SIZE];
	} stCommand;
	
	if(dwDataSize > WRITE_DATA_SIZE) return LJX8IF_RC_ERR_PARAMETER;
	
	bzero(&stCommand, sizeof(stCommand));
	memcpy(stCommand.header, header_dat, sizeof(stCommand.header));
	memcpy(stCommand.body, pData, dwDataSize);
	
	int res = myAnyCommand(lDeviceId, &stCommand.header[0], sizeof(stCommand));
	if(LJX8IF_RC_OK != res) return res;
	
	*pdwError 	= *(unsigned int*)(&myrecvline[lDeviceId][28]);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_GetSetting(int lDeviceId, unsigned char byDepth, LJX8IF_TARGET_SETTING TargetSetting, void* pData, unsigned int dwDataSize)
{
	LJX8IF_TARGET_SETTING *tset = &TargetSetting;
	
	unsigned char senddata[] = { 0x31, 0x00, 0x00, 0x00, 
								0x00, 0x00, 0x00, 0x00,
								byDepth, 0x00, 0x00, 0x00, 
								tset->byType,
								tset->byCategory,
								tset->byItem,
								0x00,
								tset->byTarget1,
								tset->byTarget2,
								tset->byTarget3,
								tset->byTarget4};
								
	int res = myAnyCommand(lDeviceId, senddata, sizeof(senddata));
	if(LJX8IF_RC_OK != res) return res;

	// Copy received setting
	memcpy(pData, &myrecvline[lDeviceId][28], dwDataSize);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_InitializeSetting(int lDeviceId, unsigned char byDepth, unsigned char byTarget)
{
	unsigned char senddata[] = { 0x3D, 0x00, 0x00, 0x00, 
								byDepth, 0x00, 0x00, 0x00, 
								0x03, byTarget, 0x00, 0x00};
	return myAnyCommand(lDeviceId, senddata, sizeof(senddata));
}

int LJX8IF_ReflectSetting(int lDeviceId, unsigned char byDepth, unsigned int* pdwError)
{
	unsigned char senddata[] = { 0x33, 0x00, 0x00, 0x00, byDepth, 0x00, 0x00, 0x00 };
	int res = myAnyCommand(lDeviceId, senddata, sizeof(senddata));
	if(LJX8IF_RC_OK != res) return res;

	*pdwError 	= *(unsigned int*)(&myrecvline[lDeviceId][28]);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_RewriteTemporarySetting(int lDeviceId, unsigned char byDepth)
{
	if(byDepth == 1 || byDepth == 2){
		byDepth -=1;
	}
	else{
		byDepth = 0xFF;
	}
	unsigned char senddata[] = { 0x35, 0x00, 0x00, 0x00, byDepth, 0x00, 0x00, 0x00 };
	return myAnyCommand(lDeviceId, senddata, sizeof(senddata));
}

int LJX8IF_CheckMemoryAccess(int lDeviceId, unsigned char* pbyBusy)
{
	int res = sendSingleCommand(lDeviceId, 0x34);
	if(LJX8IF_RC_OK != res) return res;
	
	*pbyBusy 	= *(unsigned char*)(&myrecvline[lDeviceId][28]);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_SetXpitch(int lDeviceId, unsigned int dwXpitch)
{
	unsigned char *abyDat = (unsigned char*)&dwXpitch;
	unsigned char senddata[] = { 0x36, 0x00, 0x00, 0x00, abyDat[0] , abyDat[1], abyDat[2], abyDat[3] };
	return myAnyCommand(lDeviceId, senddata, sizeof(senddata));
}

int LJX8IF_GetXpitch(int lDeviceId, unsigned int* pdwXpitch)
{
	int res = sendSingleCommand(lDeviceId, 0x37);
	if(LJX8IF_RC_OK != res) return res;
	
	*pdwXpitch 	= *(unsigned int*)(&myrecvline[lDeviceId][28]);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_SetTimerCount(int lDeviceId, unsigned int dwTimerCount)
{
	unsigned char *abyDat = (unsigned char*)&dwTimerCount;
	unsigned char senddata[] = { 0x4E, 0x00, 0x00, 0x00, abyDat[0] , abyDat[1], abyDat[2], abyDat[3] };
	return myAnyCommand(lDeviceId, senddata, sizeof(senddata));
}

int LJX8IF_GetTimerCount(int lDeviceId, unsigned int* pdwTimerCount)
{
	int res = sendSingleCommand(lDeviceId, 0x4F);
	if(LJX8IF_RC_OK != res) return res;
	
	*pdwTimerCount 	= *(unsigned int*)(&myrecvline[lDeviceId][28]);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_ChangeActiveProgram(int lDeviceId, unsigned char byProgramNo)
{
	unsigned char senddata[] = { 0x39, 0x00, 0x00, 0x00, byProgramNo, 0x00, 0x00, 0x00 };
	return myAnyCommand(lDeviceId, senddata, sizeof(senddata));
}

int LJX8IF_GetActiveProgram(int lDeviceId, unsigned char* pbyProgramNo)
{
	//There is no TCP/IP command to get the active program number directly.
	//Instead, use the "GetAttentionStatus command to get the information stored in the response.
	int res = sendSingleCommand(lDeviceId, 0x65);
	if(LJX8IF_RC_OK != res) return res;
	
	*pbyProgramNo 	= *(unsigned char*)(&myrecvline[lDeviceId][24]);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_GetProfile(int lDeviceId, LJX8IF_GET_PROFILE_REQUEST* pReq, LJX8IF_GET_PROFILE_RESPONSE* pRsp, LJX8IF_PROFILE_INFO* pProfileInfo, unsigned int* pdwProfileData, unsigned int dwDataSize)
{
	unsigned char* abyDat = (unsigned char*)(&pReq->dwGetProfileNo);
	
	unsigned char senddata[] = {0x42, 0x00, 0x00, 0x00, 
								0x00, 0x00, 0x00, 0x00,
								pReq->byTargetBank, 
								pReq->byPositionMode, 
								0x00, 0x00, 
								abyDat[0], abyDat[1], abyDat[2], abyDat[3],
								pReq->byGetProfileCount, 
								pReq->byErase, 
								0x00, 0x00};
								
	int res = myAnyCommand(lDeviceId, senddata, sizeof(senddata));
	if(LJX8IF_RC_OK != res) return res;
	
	//*pRsp
	pRsp->dwCurrentProfileNo 	= *(unsigned int*)(&myrecvline[lDeviceId][28]);
	pRsp->dwOldestProfileNo 	= *(unsigned int*)(&myrecvline[lDeviceId][32]);
	pRsp->dwGetTopProfileNo 	= *(unsigned int*)(&myrecvline[lDeviceId][36]);
	pRsp->byGetProfileCount 	= *(unsigned char*)(&myrecvline[lDeviceId][40]);
	
	//*pProfileInfo
	unsigned char kind 			= 	myrecvline[lDeviceId][48];
	pProfileInfo->byProfileCount	= ProfileDataConvert::GetProfileCount(kind);
	pProfileInfo->byLuminanceOutput = (BRIGHTNESS_VALUE & kind)>0 ? 1 : 0;
	pProfileInfo->wProfileDataCount = *(unsigned short*)(&myrecvline[lDeviceId][52]);
	unsigned short profileUnit		= *(unsigned short*)(&myrecvline[lDeviceId][54]);
	pProfileInfo->lXStart			= *(int*)(&myrecvline[lDeviceId][56]);
	pProfileInfo->lXPitch			= *(int*)(&myrecvline[lDeviceId][60]);

	//*pdwProfileData
	unsigned char* beforeConvData	= &myrecvline[lDeviceId][64];
	
	res = ProfileDataConvert::ConvertProfileData(
						pRsp->byGetProfileCount, 
						kind, 
						0, 
						pProfileInfo->wProfileDataCount,
						profileUnit,
						beforeConvData, 
						pdwProfileData, 
						dwDataSize);
						
	return res;
}

int LJX8IF_GetBatchProfile(int lDeviceId, LJX8IF_GET_BATCH_PROFILE_REQUEST* pReq, LJX8IF_GET_BATCH_PROFILE_RESPONSE* pRsp, LJX8IF_PROFILE_INFO * pProfileInfo, unsigned int* pdwBatchData, unsigned int dwDataSize)
{
	unsigned char* abyBatchNo	= (unsigned char*)(&pReq->dwGetBatchNo);
	unsigned char* abyProfNo	= (unsigned char*)(&pReq->dwGetProfileNo);
	
	unsigned char senddata[] = {0x43, 0x00, 0x00, 0x00, 
								0x00, 0x00, 0x00, 0x00,
								pReq->byTargetBank, 
								pReq->byPositionMode, 
								0x00, 0x00, 
								abyBatchNo[0], abyBatchNo[1], abyBatchNo[2], abyBatchNo[3],
								abyProfNo[0], abyProfNo[1], abyProfNo[2], abyProfNo[3],
								pReq->byGetProfileCount, 
								pReq->byErase, 
								0x00, 0x00};
								
	int res = myAnyCommand(lDeviceId, senddata, sizeof(senddata));
	if(LJX8IF_RC_OK != res) return res;
	
	//*pRsp
	pRsp->dwCurrentBatchNo 				= *(unsigned int*)(&myrecvline[lDeviceId][28]);
	pRsp->dwCurrentBatchProfileCount	= *(unsigned int*)(&myrecvline[lDeviceId][32]) & ~MASK_CURRENT_BATCH_COMMITED;
	pRsp->dwOldestBatchNo				= *(unsigned int*)(&myrecvline[lDeviceId][36]);
	pRsp->dwOldestBatchProfileCount		= *(unsigned int*)(&myrecvline[lDeviceId][40]);
	pRsp->dwGetBatchNo					= *(unsigned int*)(&myrecvline[lDeviceId][44]);
	pRsp->dwGetBatchProfileCount		= *(unsigned int*)(&myrecvline[lDeviceId][48]);
	pRsp->dwGetBatchTopProfileNo		= *(unsigned int*)(&myrecvline[lDeviceId][52]);
	pRsp->byGetProfileCount				= *(unsigned char*)(&myrecvline[lDeviceId][56]);
	pRsp->byCurrentBatchCommited		= ((*(unsigned int*)(&myrecvline[lDeviceId][32]) & MASK_CURRENT_BATCH_COMMITED) > 0) ? 1 : 0;

	//*pProfileInfo
	unsigned char kind 			= 	myrecvline[lDeviceId][48+16];
	pProfileInfo->byProfileCount	= ProfileDataConvert::GetProfileCount(kind);
	pProfileInfo->byLuminanceOutput = (BRIGHTNESS_VALUE & kind)>0 ? 1 : 0;
	pProfileInfo->wProfileDataCount = *(unsigned short*)(&myrecvline[lDeviceId][52+16]);
	unsigned short profileUnit		= *(unsigned short*)(&myrecvline[lDeviceId][54+16]);
	pProfileInfo->lXStart			= *(int*)(&myrecvline[lDeviceId][56+16]);
	pProfileInfo->lXPitch			= *(int*)(&myrecvline[lDeviceId][60+16]);

	//*pdwProfileData
	unsigned char* beforeConvData	= &myrecvline[lDeviceId][64+16];
	
	res = ProfileDataConvert::ConvertProfileData(
						pRsp->byGetProfileCount, 
						kind, 
						0, 
						pProfileInfo->wProfileDataCount,
						profileUnit,
						beforeConvData, 
						pdwBatchData, 
						dwDataSize);
						
	return res;
}

int LJX8IF_GetBatchSimpleArray(int lDeviceId, LJX8IF_GET_BATCH_PROFILE_REQUEST* pReq, LJX8IF_GET_BATCH_PROFILE_RESPONSE* pRsp, LJX8IF_PROFILE_INFO* pProfileInfo, LJX8IF_PROFILE_HEADER* pProfileHeaderArray, unsigned short* pHeightProfileArray, unsigned short* pLuminanceProfileArray)
{
	const int MaxProfileCount = 3200;
	auto bufferSize = pReq->byGetProfileCount * (MaxProfileCount * 2 + sizeof(LJX8IF_PROFILE_HEADER) + sizeof(LJX8IF_PROFILE_FOOTER));
	auto pdwBatchData = new(std::nothrow) unsigned int[bufferSize];
	if (pdwBatchData == NULL) return LJX8IF_RC_ERR_NOMEMORY;

	int res = LJX8IF_GetBatchProfile(lDeviceId, pReq, pRsp, pProfileInfo, pdwBatchData, (unsigned int)(bufferSize * sizeof(unsigned int)));
	
	if (res == LJX8IF_RC_OK) {
		unsigned short profileUnit = *(unsigned short*)(&myrecvline[lDeviceId][54+16]);

		ProfileDataConvert::ConvertProfileData32to16AsSimpleArray(
			(int*)pdwBatchData,
			pProfileInfo->wProfileDataCount,
			pProfileInfo->byLuminanceOutput == 1,
			(unsigned short)pRsp->byGetProfileCount,
			profileUnit,
			pProfileHeaderArray,
			pHeightProfileArray,
			pLuminanceProfileArray);
	}
	
	delete [] pdwBatchData;
	pdwBatchData = NULL;
	return res;
}

int LJX8IF_PreStartHighSpeedDataCommunication(int lDeviceId, LJX8IF_HIGH_SPEED_PRE_START_REQ* pReq, LJX8IF_PROFILE_INFO* pProfileInfo)
{
	unsigned char profileKind;
	unsigned short profileUnit;

	unsigned char senddata[] = { 0x47, 0x00, 0x00, 0x00, (pReq->bySendPosition), 0x00, 0x00, 0x00 };
	int res = myAnyCommand(lDeviceId, senddata, sizeof(senddata));
	if(LJX8IF_RC_OK != res) return res;

	startcode[lDeviceId] = *(int*)(&myrecvline[lDeviceId][32]);
	profileKind = myrecvline[lDeviceId][40];
	
	pProfileInfo->byProfileCount	= 1;
	pProfileInfo->wProfileDataCount = *(unsigned short*)(&myrecvline[lDeviceId][44]);
	profileUnit 					= *(unsigned short*)(&myrecvline[lDeviceId][46]);
	
	pProfileInfo->byLuminanceOutput = (BRIGHTNESS_VALUE & profileKind)>0 ? 1 : 0;
	
	pProfileInfo->lXStart 			= *(int*)(&myrecvline[lDeviceId][48]);
	pProfileInfo->lXPitch			= *(int*)(&myrecvline[lDeviceId][52]);

	SetThreadParamFast(lDeviceId, pProfileInfo->wProfileDataCount, profileKind, profileUnit);
	
	return LJX8IF_RC_OK;
}

int LJX8IF_StartHighSpeedDataCommunication(int lDeviceId)
{
	unsigned char *abyDat = (unsigned char*)&startcode[lDeviceId];
	unsigned char senddata[] = { 0xA0, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00, abyDat[0], abyDat[1], abyDat[2], abyDat[3] };
	return myAnyCommand(lDeviceId, senddata, sizeof(senddata));
}

int LJX8IF_StopHighSpeedDataCommunication(int lDeviceId)
{
	return sendSingleCommand(lDeviceId, 0x48);
}

int LJX8IF_InitializeHighSpeedDataCommunication(int lDeviceId, LJX8IF_ETHERNET_CONFIG* pEthernetConfig, unsigned short wHighSpeedPortNo, LJX8IF_CALLBACK pCallBack, unsigned int dwProfileCount, unsigned int dwThreadId)
{
	char ip[15];
	unsigned char *ipn;	
	struct sockaddr_in servaddr;
	
	int rc = LJX8IF_RC_OK;

	if(pEthernetConfig == NULL)
		return LJX8IF_RC_ERR_PARAMETER;
	
	ipn= pEthernetConfig->abyIpAddress;
	
	sprintf(ip,"%d.%d.%d.%d",ipn[0],ipn[1],ipn[2],ipn[3]);
	sockfdHighSpeed[lDeviceId] = socket(AF_INET,SOCK_STREAM,0);
	if (sockfdHighSpeed[lDeviceId] < 0) {
		return LJX8IF_RC_ERR_OPEN;
	}

	linger linger_opt;
	linger_opt.l_onoff = 1;
	linger_opt.l_linger = 1;

	if (setsockopt(sockfdHighSpeed[lDeviceId], SOL_SOCKET, SO_LINGER, reinterpret_cast<char*>(&linger_opt), sizeof(linger_opt)) < 0) {
		close(sockfdHighSpeed[lDeviceId]);
		return LJX8IF_RC_ERR_OPEN;
	}

	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = inet_addr(ip);
	servaddr.sin_port = htons(wHighSpeedPortNo);

	// Connect socket in non-blocking mode to determine timeout
	// set non-blocking mode
	int flag;
	if ((flag = fcntl(sockfdHighSpeed[lDeviceId], F_GETFL, NULL)) < 0) {
		return LJX8IF_RC_ERR_OPEN;
	}
	if (fcntl(sockfdHighSpeed[lDeviceId], F_SETFL, flag | O_NONBLOCK) < 0) {
		return LJX8IF_RC_ERR_OPEN;
	}

	if (connect(sockfdHighSpeed[lDeviceId], (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
		if (errno != EINPROGRESS) {
			return LJX8IF_RC_ERR_OPEN;
		}

		struct timeval tv;
		int res;
		fd_set fds;

		tv.tv_sec = LJXA_CONNECT_TIMEOUT_SEC;
		tv.tv_usec = 0;
		FD_ZERO(&fds);
		FD_SET(sockfdHighSpeed[lDeviceId], &fds);

		res = select(sockfdHighSpeed[lDeviceId] + 1, NULL, &fds, NULL, &tv);

		if (res <= 0) { // Connection timeout
			return LJX8IF_RC_ERR_OPEN;
		}
		else if (res > 0) {
			socklen_t optlen = sizeof(servaddr);

			// Check if socket is writable
			if (getpeername(sockfdHighSpeed[lDeviceId], (struct sockaddr*)&servaddr, &optlen) < 0) {
				return LJX8IF_RC_ERR_OPEN;
			}
		}
	}

	// Blocking mode
	if (fcntl(sockfdHighSpeed[lDeviceId], F_SETFL, flag) < 0) {
		return LJX8IF_RC_ERR_OPEN;
	}
	
	InitThreadParamFast(lDeviceId);
	
	SetThreadInitParamFast(lDeviceId, pCallBack , dwProfileCount , dwThreadId);

	do {
		//Allocate buffer for raw data
		if (m_ThreadParamFast[lDeviceId].pbyTempBuffer == NULL) {
			m_ThreadParamFast[lDeviceId].pbyTempBuffer = new(std::nothrow) unsigned char[FAST_RECEIVE_BUF_SIZE];
			if (m_ThreadParamFast[lDeviceId].pbyTempBuffer == NULL) {
				rc = LJX8IF_RC_ERR_NOMEMORY;
				break;
			}
		}
		//Allocate buffer for 20bit data
		if (m_ThreadParamFast[lDeviceId].pbyConvBuffer == NULL) {
			m_ThreadParamFast[lDeviceId].pbyConvBuffer = new(std::nothrow) unsigned char[FAST_RECEIVE_BUF_SIZE*2];
			if (m_ThreadParamFast[lDeviceId].pbyConvBuffer == NULL) {
				rc = LJX8IF_RC_ERR_NOMEMORY;
				break;
			}
		}
		//Allocate buffer for 32bit data
		if (m_ThreadParamFast[lDeviceId].pbyConvEndBuffer == NULL) {
			m_ThreadParamFast[lDeviceId].pbyConvEndBuffer = new(std::nothrow) unsigned char[FAST_RECEIVE_BUF_SIZE];
			if (m_ThreadParamFast[lDeviceId].pbyConvEndBuffer == NULL) {
				rc = LJX8IF_RC_ERR_NOMEMORY;
				break;
			}
		}
		
		//Create a thread for receiving high-speed data
		m_ThreadParamFast[lDeviceId].stopFlag = 0;
		int res = pthread_create(&pthread[lDeviceId], NULL, &receiveThread, (void*)&m_ThreadParamFast[lDeviceId]);
		usleep(1000);
		if (res != 0) {
			rc = LJX8IF_RC_ERR_OPEN;
			break;
		}
		else {
			// Wait for the thread to start (max 100ms)
			rc = LJX8IF_RC_ERR_OPEN;
			for (int i = 0; i < 10; i++) {
				if (m_ThreadParamFast[lDeviceId].bIsEnabled) {
					rc = LJX8IF_RC_OK;
					break;
				}
				usleep(10000);
			}
			if (rc != LJX8IF_RC_OK) {
				// If an error occurs, terminate the thread
				if (pthread[lDeviceId] != 0) {
					pthread_cancel(pthread[lDeviceId]);
					pthread[lDeviceId] = 0;
				}
			}
		}
		
	}while(0);
	
	return 	(int)rc;	
}

int LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray(int lDeviceId, LJX8IF_ETHERNET_CONFIG* pEthernetConfig, unsigned short wHighSpeedPortNo, LJX8IF_CALLBACK_SIMPLE_ARRAY pCallBackSimpleArray, unsigned int dwProfileCount, unsigned int dwThreadId)
{
	
	int rc = LJX8IF_RC_OK;
	
	m_ThreadParamFast[lDeviceId].pFuncSimpleArray = pCallBackSimpleArray;
	
	rc = LJX8IF_InitializeHighSpeedDataCommunication(lDeviceId, pEthernetConfig, wHighSpeedPortNo, &CallbackFuncWrapper, dwProfileCount, dwThreadId);
	
	return 	(int)rc;
}

int LJX8IF_FinalizeHighSpeedDataCommunication(int lDeviceId)
{	
	//Stop high-speed communication
	LJX8IF_StopHighSpeedDataCommunication(lDeviceId);
	
	//Terminate the thread for receiving high-speed data
	m_ThreadParamFast[lDeviceId].stopFlag = 1;
	pthread_join( pthread[lDeviceId], NULL);
	
    //Release buffer
    if (m_ThreadParamFast[lDeviceId].pbyTempBuffer != NULL) {
        delete[] m_ThreadParamFast[lDeviceId].pbyTempBuffer;
        m_ThreadParamFast[lDeviceId].pbyTempBuffer = NULL;
    }
    if (m_ThreadParamFast[lDeviceId].pbyConvBuffer != NULL) {
        delete[] m_ThreadParamFast[lDeviceId].pbyConvBuffer;
        m_ThreadParamFast[lDeviceId].pbyConvBuffer = NULL;
    }
    if (m_ThreadParamFast[lDeviceId].pbyConvEndBuffer != NULL) {
        delete[] m_ThreadParamFast[lDeviceId].pbyConvEndBuffer;
        m_ThreadParamFast[lDeviceId].pbyConvEndBuffer = NULL;
    }

	return LJX8IF_RC_OK;
}

int LJX8IF_GetZUnitSimpleArray(int lDeviceId, unsigned short* pwZUnit)
{
	if (m_ThreadParamFast[lDeviceId].wUnit == 0) return 0x80A0;
	*pwZUnit = m_ThreadParamFast[lDeviceId].wUnit * 8;
	return LJX8IF_RC_OK;
}

}//extern "C"


//-----------------------------------------------------------------
// Internal funcitons implementation
//-----------------------------------------------------------------
static void *receiveThread(void *args)
{	
	THREADPARAM_FAST *pThreadParam = (THREADPARAM_FAST *) args;
	int lDeviceId = pThreadParam->dwDeviceId;
	pThreadParam->bIsEnabled = 1;
	
	int m_nNumData = 0;
	int m_nWriteIndex =0;
	
	int dwNumReceived;
	
	//for select
	struct timeval tv;
	fd_set fds, readfds;
	FD_ZERO(&readfds);
	FD_SET(sockfdHighSpeed[lDeviceId],&readfds);

	while(1){ //loop1
		if(pThreadParam->stopFlag){
			break;
		}
		
		// Use select for non-blocking recv
		tv.tv_sec = 1;
		tv.tv_usec = 0;
		memcpy(&fds, &readfds, sizeof(fd_set));
		select(sockfdHighSpeed[lDeviceId]+1, &fds, NULL, NULL, &tv);
		dwNumReceived = 0;

		if( FD_ISSET(sockfdHighSpeed[lDeviceId], &fds)){
			dwNumReceived = recv(sockfdHighSpeed[lDeviceId],(unsigned char*)pThreadParam->pbyTempBuffer, FAST_RECEIVE_BUF_SIZE, 0);
		}
		
		if(pThreadParam->stopFlag){
			break;
		}
		
		// Skip if buffer is null
		if (pThreadParam->dwProfileSize32 == 0) {
			continue;
		}
		
		// Receive some data,
		if (dwNumReceived > 0) {
			// Copy data from newly received buffer to 20bit profile buffer.
			memcpy(pThreadParam->pbyConvBuffer + pThreadParam->dwConvBufferIndex, pThreadParam->pbyTempBuffer, dwNumReceived);
			pThreadParam->dwConvBufferIndex += dwNumReceived;
			unsigned char* pbyConvBefore = (unsigned char*)pThreadParam->pbyConvBuffer;
			unsigned int dwNotify = 0;
			int rc = 1;
			
			while (1) { //loop2
				while (1) { //loop3
					// Check the received data, and convert 20bit data to 32bit data.
					unsigned int dwConvSize = 0;
					unsigned int dwNumUsed = ReceiveDataCheck(pbyConvBefore, pThreadParam->dwConvBufferIndex, pThreadParam->dwProfileSize,
											pThreadParam->byKind, pThreadParam->dwProfileCnt, pThreadParam->wUnit, (unsigned char*)pThreadParam->pbyConvEndBuffer, &dwConvSize, &dwNotify
											, dwNumReceived, pThreadParam->pbyTempBuffer);
					
					// Received data lacks. Wait for next data.
					if (dwNumUsed == 0) {
						break;
					}
					// In case of "notify" packet, notify upper function.
					// if you get parity error, read all.
					if ( (dwNotify & ~PARITY_ERROR ) != 0) {
						pThreadParam->dwConvBufferIndex -= dwNumUsed;
						pbyConvBefore += dwNumUsed;
						break;
					
					// In case of profile data.
					} else {
						
						// The requested number of profiles has been collected.
						if (pThreadParam->dwProfileLoopCnt >= pThreadParam->dwProfileMax) {
							break;
						}
						
						int nSizeOfData = dwConvSize;
						unsigned char *m_pBuffer = pThreadParam->ReceiveBuffer;
						unsigned char *pbyCopyData = pThreadParam->pbyConvEndBuffer;
						
						// Check if the buffer size is not exceeded.
						if (m_nNumData+nSizeOfData <= pThreadParam->nBufferSize) {
							if (m_nWriteIndex+nSizeOfData > pThreadParam->nBufferSize) {
								memcpy(m_pBuffer+m_nWriteIndex,pbyCopyData,pThreadParam->nBufferSize-m_nWriteIndex);
								pbyCopyData += (pThreadParam->nBufferSize-m_nWriteIndex);
								nSizeOfData -= (pThreadParam->nBufferSize-m_nWriteIndex);
								m_nNumData += (pThreadParam->nBufferSize-m_nWriteIndex);
								m_nWriteIndex = 0;
							}
							if (nSizeOfData > 0) {
								memcpy(m_pBuffer+m_nWriteIndex,pbyCopyData,nSizeOfData);
								m_nWriteIndex = (m_nWriteIndex+nSizeOfData) % pThreadParam->nBufferSize;
								m_nNumData += nSizeOfData;
							}
							rc = 1;
						}
						else {	// Irregular case. The buffer size is exceeded.
							rc = 0;
							break;
						}
						
						pThreadParam->dwConvBufferIndex -= dwNumUsed;
						pbyConvBefore += dwNumUsed;
						pThreadParam->dwProfileLoopCnt ++;
					}
				} //loop3
				
				// Received some data. Wake up the callback function.
				if (rc) {
					// If profile data is received and there is remaining received data, pad them in the conversion buffer. 
					if ((pThreadParam->dwConvBufferIndex) && (pThreadParam->pbyConvBuffer != pbyConvBefore)) {
						memcpy(pThreadParam->pbyConvBuffer, pbyConvBefore, pThreadParam->dwConvBufferIndex);
						pbyConvBefore = pThreadParam->pbyConvBuffer;
					}
					
					// "Notify" packet or the requested number of profiles has been collected.
					if ((dwNotify != 0) || (pThreadParam->dwProfileLoopCnt >= pThreadParam->dwProfileMax)) {
						
						unsigned char* pbyBuffer = pThreadParam->ReceiveBuffer;
						pThreadParam->pFunc(pbyBuffer, pThreadParam->dwProfileSize32, pThreadParam->dwProfileLoopCnt, dwNotify, pThreadParam->dwThreadId);
						
						m_nNumData = 0;
						m_nWriteIndex =0;
						pThreadParam->dwProfileLoopCnt = 0;
						
						// "Notify" packet and "stop continuous" transmission
						if (dwNotify & 0x0000ffff) {
							// Free buffer and clear data.
							if (pThreadParam->ReceiveBuffer != NULL) {
								free(pThreadParam->ReceiveBuffer);
								pThreadParam->ReceiveBuffer = NULL;
							}
							pThreadParam->dwProfileSize32 = 0;
							pThreadParam->dwConvBufferIndex = 0;
							break;
						}
					// Received data lacks. Wait for next data.
					} else {
						break;
					}
					
				// Irregular case. The buffer size is exceeded.
				} else {
					break;
				}
			}//loop2
			
			if (!rc) {
				// Irregular case. The buffer size is exceeded or parity error.
				break;
			}
		}
	}//loop1
	
	//Terminate thread
	pThreadParam->bIsEnabled = 0;
	
	pthread_exit(0);
}

static int sendSingleCommand(int lDeviceId, unsigned char CommandCode)
{
	unsigned char senddata[] = { CommandCode, 0x00, 0x00, 0x00 };
	return myAnyCommand(lDeviceId, senddata, sizeof(senddata));
}

static int myAnyCommand(int lDeviceId, unsigned char* senddata, int dataLength)
{
	int dataOffset = 16;
	int tcpLength = dataOffset + dataLength -4;
	int *lengthPtr;
	
	lengthPtr = (int*)(&mysendBuf[lDeviceId][0]);
	*lengthPtr = tcpLength;
	
	mysendBuf[lDeviceId][4] = 0x03;
	mysendBuf[lDeviceId][5] = 0x00;
	mysendBuf[lDeviceId][6] = 0xF0;
	mysendBuf[lDeviceId][7] = 0x00;
		
	lengthPtr = (int*)(&mysendBuf[lDeviceId][12]);
	*lengthPtr = dataLength;

	// Clear receive buffer before send data.
	{
		// for select
		struct timeval tv;
		fd_set fds, readfds;
		FD_ZERO(&readfds);
		FD_SET(sockfd[lDeviceId],&readfds);

		// use select for non-blocking recv
		tv.tv_sec = 0;
		tv.tv_usec = 1;
		memcpy(&fds, &readfds, sizeof(fd_set));
		select(sockfd[lDeviceId]+1, &fds, NULL, NULL, &tv);

		if( FD_ISSET(sockfd[lDeviceId], &fds)){
			recv(sockfd[lDeviceId],&myrecvline[lDeviceId][0],RECEIVE_BUF_SIZE, 0);
		}
		
		//clear buffer
		bzero(&myrecvline[lDeviceId][0],RECEIVE_BUF_SIZE);
	}


	// Send
	memcpy(&mysendBuf[lDeviceId][0] + dataOffset, senddata, dataLength);
	auto res = send(sockfd[lDeviceId], &mysendBuf[lDeviceId][0], tcpLength + 4, MSG_DONTROUTE); //send command
	if(res == -1){
		return LJX8IF_RC_ERR_SEND;
	}
	
	// Check if the minimum data has arrived
	int n = 0;
	while( n < 4 ){
		n = recv(sockfd[lDeviceId], &myrecvline[lDeviceId][0], RECEIVE_BUF_SIZE, MSG_PEEK);
		if(n==-1){
			return LJX8IF_RC_ERR_RECEIVE;
		}
	}
	// Receive rest of the data
	int recvLength = *(int *)&myrecvline[lDeviceId][0];
	int targetLength = recvLength + 4 ;
	int tryBytes	 = 0;
	int receivedBytes = 0;
	
	// Receivable size at one time is limited due to the recv function.
	// Divide and receive repeatedly.
	while( receivedBytes < targetLength ){
		
		tryBytes = targetLength - receivedBytes;
		if(tryBytes > 65535)
		{
			tryBytes = 65535;
		}
		
		n = 0;
		while( n < tryBytes ){
			n = recv(sockfd[lDeviceId], &myrecvline[lDeviceId][receivedBytes], tryBytes, MSG_PEEK);
			if(n==-1){
				return LJX8IF_RC_ERR_RECEIVE;
			}
		}
		
		recv(sockfd[lDeviceId], &myrecvline[lDeviceId][receivedBytes], tryBytes, MSG_WAITALL);
		receivedBytes += tryBytes;
		
		//printf("<%u>/<%u>\n",receivedBytes,targetLength);
	}
	myrecvline[lDeviceId][targetLength]=0; //add string termination character
	
	// Check response code
	unsigned char errCode= myrecvline[lDeviceId][RES_ADRESS_OFFSET_ERROR];

	if (errCode != 0){
		return ( PREFIX_ERROR_CONTROLLER + errCode );
	}
	return LJX8IF_RC_OK;
}

/**
  @brief	Checking received data. Used in high speed receiving thread. 
  @param	pbyConvBeforeData	: 20bit profile data (BEFORE conversion)
  @param	dwRecvSize			: cumlative value of data size for 20bit profiles (in byte)
  @param	dwProfSize			: data size for each 20bit profile (in byte)
  @param	byKind				: kind of profile (e.g. luminance output is enabled)
  @param	dwProfCnt			: number of points for each profile
  @param	wUnit				: unit of profile(in 0.01um)
  
  @param	pbyConvAfterData	: 32bit profile data (AFTER conversion)
  @param	pdwConvSize			: data size for each 32bit profile (in byte)
  
  @param	pdwNotify			: "notify" content
  @param	dwNumReceived		: newly received data size for 20bit profiles (in byte)
  @param	pbyTempBuffer		: newly received data buffer
  @return	used size
*/

static unsigned int ConvProfData20to32(unsigned char* pbyConvBeforeData, unsigned char byKind, unsigned int dwProfCnt, unsigned short wUnit, unsigned char* pbyConvAfterData, int *bIsRecvProfErr, unsigned int dwProfSize)
{
	unsigned int dwConvSize = 0;
	if ((pbyConvBeforeData == NULL) || (pbyConvAfterData == NULL)) {
		return dwConvSize;
	}

	int *zprof = (int*)(pbyConvAfterData + PROF_HEADER_LEN);

	memcpy(pbyConvAfterData,pbyConvBeforeData,sizeof(int)*PROF_HEADER_LEN);	
	
	ProfileDataConvert::ConvertProfileData20to32(byKind, dwProfCnt, wUnit, pbyConvBeforeData, zprof);

	// Check parity
	unsigned int dwCalcParity = 0;
	unsigned int* pdwTmpRecvData = (unsigned int*)pbyConvBeforeData;
	unsigned int dwOneProfDataCnt = (dwProfSize - PROF_CHECKSUM_LEN) / sizeof(unsigned int);
	for (unsigned int i = 0; i < dwOneProfDataCnt; i++) {
		dwCalcParity += *pdwTmpRecvData;
		pdwTmpRecvData++;
	}
	*bIsRecvProfErr = (*pdwTmpRecvData != dwCalcParity);

	int multipleValue = (byKind & BRIGHTNESS_VALUE) == BRIGHTNESS_VALUE ? 2 : 1;
	dwConvSize = (dwProfCnt * multipleValue * sizeof(unsigned int)) + PROF_HEADER_LEN + PROF_CHECKSUM_LEN;

	return dwConvSize;
}

/**
  @brief	Checking received data. Used in high speed receiving thread. 
  @param	pbyConvBeforeData	: 20bit profile data (BEFORE conversion)
  @param	dwRecvSize			: cumlative value of data size for 20bit profiles (in byte)
  @param	dwProfSize			: data size for each 20bit profile (in byte)
  @param	byKind				: kind of profile (e.g. luminance output is enabled)
  @param	dwProfCnt			: number of points for each profile
  @param	wUnit				: unit of profile(in 0.01um)
  
  @param	pbyConvAfterData	: 32bit profile data (AFTER conversion)
  @param	pdwConvSize			: data size for each 32bit profile (in byte)
  
  @param	pdwNotify			: "notify" content
  @param	dwNumReceived		: newly received data size for 20bit profiles (in byte)
  @param	pbyTempBuffer		: newly received data buffer
  @return	used size
*/
static unsigned int ReceiveDataCheck(unsigned char* pbyConvBeforeData, unsigned int dwRecvSize, unsigned int dwProfSize, unsigned char byKind, unsigned int dwProfCnt, unsigned short wUnit, unsigned char* pbyConvAfterData, unsigned int* pdwConvSize, unsigned int* pdwNotify, unsigned int dwNumReceived, unsigned char* pbyTempBuffer)
{
	ST_CIND_PROFILE_NOTIFY* pstNotify = (ST_CIND_PROFILE_NOTIFY*)pbyConvBeforeData;
	unsigned int dwUsedSize = 0;
	unsigned int dwNotify = 0;
	unsigned int dwConvSize = 0;

	if ((pbyConvBeforeData == NULL) || (pbyConvAfterData == NULL) || (pdwConvSize == NULL) || (pdwNotify == NULL)) {
		return dwUsedSize;
	}

	do {
		// Check if "Notify" packet.
		if (dwRecvSize >= sizeof(ST_CIND_PROFILE_NOTIFY)) {
			if (pstNotify->dwMagicNo == DEF_NOTIFY_MAGIC_NO) {
				dwNotify = pstNotify->dwNotify;
				dwUsedSize = sizeof(ST_CIND_PROFILE_NOTIFY);
				break;
			}
		}
		
		// Convert profile data (20bit to 32bit)
		if (dwRecvSize >= dwProfSize) {
			int bIsRecvProfErr = 0;
			dwConvSize = ConvProfData20to32(pbyConvBeforeData, byKind, dwProfCnt, wUnit, pbyConvAfterData, &bIsRecvProfErr, dwProfSize);
			if (bIsRecvProfErr) {
				dwNotify |= PARITY_ERROR;
			}
			dwUsedSize = dwProfSize;
			break;
		}
		
		// Irregular case.
		if (dwNumReceived == sizeof(ST_CIND_PROFILE_NOTIFY) && dwRecvSize > 0) {
			pstNotify = (ST_CIND_PROFILE_NOTIFY*)pbyTempBuffer;
			
			if (pstNotify->dwMagicNo == DEF_NOTIFY_MAGIC_NO && (pstNotify->dwNotify & 0x0000ffff) != 0) {	// "Notify" packet and Stop continuous transmission
				dwNotify = pstNotify->dwNotify;
				dwUsedSize = sizeof(ST_CIND_PROFILE_NOTIFY);
				break;
			}
		}

	} while (0);

	// Update "Notify" content
	*pdwNotify = dwNotify;
	// Update the data size of each profile.(e.g. 32bit converted profiles)
	*pdwConvSize = dwConvSize;

	return dwUsedSize;
}



/**
Setting initial parameters for highspeed receiving thread
  @param	lDeviceId		: device identifier
  @param	pCallBack		: receive completion callback function
  @param	dwCnt			: number of lines to wake receive completion callback function
  @param	dwThreadId		: thread identifier
  @return	nothing
*/
void SetThreadInitParamFast(int lDeviceId, void (*pCallBack)(unsigned char*, unsigned int, unsigned int, unsigned int, unsigned int), unsigned int dwCnt, unsigned int dwThreadId)
{
	m_ThreadParamFast[lDeviceId].pFunc	= pCallBack;
	m_ThreadParamFast[lDeviceId].dwProfileMax = dwCnt;
	m_ThreadParamFast[lDeviceId].dwThreadId = dwThreadId;
	m_ThreadParamFast[lDeviceId].dwDeviceId = lDeviceId;
}

/**
Initialize parameters for highspeed receiving thread
  @param	lDeviceId		: device identifier
  @return	nothing
*/
void InitThreadParamFast(int lDeviceId)
{
	//m_ThreadParamFast[lDeviceId].ReceiveBuffer;	//Allocate in SetThreadParamFat funtion.
	
	m_ThreadParamFast[lDeviceId].pbyTempBuffer = NULL;
	m_ThreadParamFast[lDeviceId].pbyConvBuffer = NULL;
	m_ThreadParamFast[lDeviceId].dwConvBufferIndex = 0;
	m_ThreadParamFast[lDeviceId].pbyConvEndBuffer = NULL;
	m_ThreadParamFast[lDeviceId].bIsEnabled	= 0;

	m_ThreadParamFast[lDeviceId].pFunc	= NULL;
	
	m_ThreadParamFast[lDeviceId].dwProfileMax = 0;
	m_ThreadParamFast[lDeviceId].dwProfileLoopCnt = 0;
	m_ThreadParamFast[lDeviceId].dwProfileCnt = 0;
	m_ThreadParamFast[lDeviceId].dwProfileSize = 0;
	m_ThreadParamFast[lDeviceId].dwProfileSize32 = 0;
	m_ThreadParamFast[lDeviceId].wUnit = 0;
	m_ThreadParamFast[lDeviceId].SimpleArrayZUnit_um = 0;
	m_ThreadParamFast[lDeviceId].byKind = 0;
	
	m_ThreadParamFast[lDeviceId].dwThreadId = 0;
	m_ThreadParamFast[lDeviceId].dwDeviceId = 0;
	
	m_ThreadParamFast[lDeviceId].nBufferSize = 0;
	m_ThreadParamFast[lDeviceId].stopFlag = 1;
}

/**
Setting parameters for highspeed receiving thread
  @param	lDeviceId		: device identifier
  @param	dwProfileCnt	: number of data points for each profile
  @param	byKind			: kind of profile (e.g. luminance output is enabled)
  @param	wUnit			: unit of profile(in 0.01um)
  @return
*/
int SetThreadParamFast(int lDeviceId, unsigned int dwProfileCnt, unsigned char byKind, unsigned short wUnit)
{
	// Check if high speed receiving thread is enabled.
	if (! m_ThreadParamFast[lDeviceId].bIsEnabled) {
		return LJX8IF_RC_ERR_NOT_OPEN;
	}

	// Store the unit of profile.
	m_ThreadParamFast[lDeviceId].wUnit = wUnit;
	m_ThreadParamFast[lDeviceId].SimpleArrayZUnit_um = (float)(wUnit * 8 / 100.0);

	// Store the kind of profile.
	m_ThreadParamFast[lDeviceId].byKind = byKind;

	// Store the data points for each profile.
	m_ThreadParamFast[lDeviceId].dwProfileCnt = dwProfileCnt;

	// Store the size of each profile(in 20bit or 10bit raw level profile)
	// Calculate the size of profile depending on luminance output is enabled.
	unsigned int dwSize = 0;
	int isBrightness = (byKind & BRIGHTNESS_VALUE) == BRIGHTNESS_VALUE;
	if (isBrightness) {
		dwSize = divRoundUp((dwProfileCnt * 20), 32) + divRoundUp((dwProfileCnt * 10), 32);
	}
	else {
		dwSize = divRoundUp((dwProfileCnt * 20), 32);
	}
	dwSize = (dwSize * sizeof(unsigned int)) + PROF_HEADER_LEN + PROF_CHECKSUM_LEN;
	
	m_ThreadParamFast[lDeviceId].dwProfileSize = dwSize;

	// Store the size of each profile(in 32bit profile)
	// Calculate the size of profile depending on luminance output is enabled.
	dwSize = dwProfileCnt;
	if (isBrightness) {
		dwSize *= 2;
	}
	dwSize = (dwSize * sizeof(unsigned int)) + PROF_HEADER_LEN + PROF_CHECKSUM_LEN;
	m_ThreadParamFast[lDeviceId].dwProfileSize32 = dwSize;

	// Create buffer for receive completion callback function.
	if(m_ThreadParamFast[lDeviceId].ReceiveBuffer != NULL) {
		free(m_ThreadParamFast[lDeviceId].ReceiveBuffer);
        m_ThreadParamFast[lDeviceId].ReceiveBuffer = NULL;
    }
		
	m_ThreadParamFast[lDeviceId].nBufferSize  = dwSize * m_ThreadParamFast[lDeviceId].dwProfileMax;
	m_ThreadParamFast[lDeviceId].ReceiveBuffer = (unsigned char*)malloc(dwSize * m_ThreadParamFast[lDeviceId].dwProfileMax);
	
	if (m_ThreadParamFast[lDeviceId].ReceiveBuffer == NULL) {
		return LJX8IF_RC_ERR_NOMEMORY;
	}

	return LJX8IF_RC_OK;
}

//LJXA_INTERNAL_PARAM getInternalParam(int lDeviceId)
//{
//	LJXA_INTERNAL_PARAM param;
//	param.SimpleArrayZUnit_um = m_ThreadParamFast[lDeviceId].SimpleArrayZUnit_um;
//	
//	return (param);
//}

static void CallbackFuncWrapper( unsigned char *Buf, unsigned int dwSize ,unsigned int dwCount, unsigned int dwNotify, unsigned int  dwUser)
{
	unsigned int lDeviceId = dwUser;

	auto header = new LJX8IF_PROFILE_HEADER[dwCount];
	auto profile = new unsigned short[m_ThreadParamFast[lDeviceId].dwProfileCnt * dwCount];
	auto luminance = new unsigned short[m_ThreadParamFast[lDeviceId].dwProfileCnt* dwCount];
	
	int isBrightness = (m_ThreadParamFast[lDeviceId].byKind & BRIGHTNESS_VALUE) == BRIGHTNESS_VALUE;

	ProfileDataConvert::ConvertProfileData32to16AsSimpleArray(
		(int*)Buf,
		m_ThreadParamFast[lDeviceId].dwProfileCnt,
		isBrightness,
		dwCount,
		m_ThreadParamFast[lDeviceId].wUnit,
		header,
		profile,
		luminance);
		
	m_ThreadParamFast[lDeviceId].pFuncSimpleArray(header, profile, luminance, (unsigned int)isBrightness, m_ThreadParamFast[lDeviceId].dwProfileCnt, dwCount, dwNotify, dwUser);

	delete[] header;
	delete[] profile;
	delete[] luminance;
	
	header = NULL;
	profile = NULL;
	luminance = NULL;
}
