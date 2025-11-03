//Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.
/** @file
@brief	LJXA Image Acquisition Library Implementation
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#include "LJX8_IF_Linux.h"
#include "LJX8_ErrorCode.h"

#include "LJXA_ACQ.h"

// Static variable
static LJX8IF_ETHERNET_CONFIG _ethernetConfig[ MAX_LJXA_DEVICENUM ];
static int _highSpeedPortNo[ MAX_LJXA_DEVICENUM ];
static int _imageAvailable[ MAX_LJXA_DEVICENUM ];
static int _lastImageSizeHeight[ MAX_LJXA_DEVICENUM ];
static LJXA_ACQ_GETPARAM _getParam[ MAX_LJXA_DEVICENUM ];
static char *_dataBuf[ MAX_LJXA_DEVICENUM ];

// Function prototype
static void myCallbackFunc( unsigned char *Buf, unsigned int dwSize ,unsigned int dwCount, unsigned int dwNotify, unsigned int  dwUser);

int LJXA_ACQ_OpenDevice(int lDeviceId, LJX8IF_ETHERNET_CONFIG *EthernetConfig, int HighSpeedPortNo){
	int errCode = LJX8IF_EthernetOpen(lDeviceId,EthernetConfig);
	
	_ethernetConfig[ lDeviceId ] = *EthernetConfig;
	_highSpeedPortNo[ lDeviceId ] = HighSpeedPortNo;
	printf("[@(LJXA_ACQ_OpenDevice) Open device](0x%x)\n",errCode);
	
	return errCode;
}

void LJXA_ACQ_CloseDevice(int lDeviceId){
	LJX8IF_FinalizeHighSpeedDataCommunication(lDeviceId);
	LJX8IF_CommunicationClose(lDeviceId);
	printf("[@(LJXA_ACQ_CloseDevice) Close device]\n");
}

int LJXA_ACQ_Acquire(int lDeviceId, unsigned short *heightImage, unsigned char *luminanceImage, LJXA_ACQ_SETPARAM *setParam, LJXA_ACQ_GETPARAM *getParam){
	int errCode;
	
	int yDataNum = setParam->y_linenum;
	int timeout_ms = setParam->timeout_ms;
	int use_external_batchStart = setParam->use_external_batchStart;
	
	//Initialize
	errCode = LJX8IF_InitializeHighSpeedDataCommunication(lDeviceId, &_ethernetConfig[lDeviceId], _highSpeedPortNo[lDeviceId], &myCallbackFunc, yDataNum, lDeviceId);
	printf("[@(LJXA_ACQ_Acquire) Initialize HighSpeed](0x%x)\n",errCode);
	if (errCode != LJX8IF_RC_OK) return errCode;
	
	//PreStart
	LJX8IF_HIGH_SPEED_PRE_START_REQ startReq;
	startReq.bySendPosition = 2;
	LJX8IF_PROFILE_INFO profileInfo;
	
	errCode = LJX8IF_PreStartHighSpeedDataCommunication(lDeviceId, &startReq, &profileInfo);
	printf("[@(LJXA_ACQ_Acquire) PreStart](0x%x)\n",errCode);
	if (errCode != LJX8IF_RC_OK) return errCode;
	
	//Allocate memory
	_dataBuf[lDeviceId] = (char*)malloc(yDataNum * (MAX_LJXA_XDATANUM * 2 + 7) * sizeof(int));
	if (_dataBuf[lDeviceId] == NULL) {
		return LJX8IF_RC_ERR_NOMEMORY;
	}

	//Start HighSpeed
	_imageAvailable[lDeviceId] = 0;
	_lastImageSizeHeight[ lDeviceId ] = 0;
	
	errCode = LJX8IF_StartHighSpeedDataCommunication(lDeviceId);
	printf("[@(LJXA_ACQ_Acquire) Start HighSpeed](0x%x)\n",errCode);
	if (errCode != LJX8IF_RC_OK) {
		//Free memory
		if (_dataBuf[lDeviceId] != NULL) {
			free(_dataBuf[lDeviceId]);
			_dataBuf[lDeviceId] = NULL;
		}
		return errCode;
	}
	
	//StartMeasure(Batch Start)
	if ( use_external_batchStart > 0 ){
	}
	else{
		errCode = LJX8IF_StartMeasure(lDeviceId);
		printf("[@(LJXA_ACQ_Acquire) Measure Start(Batch Start)](0x%x)\n",errCode);
	}

	printf(" [@(LJXA_ACQ_Acquire) acquring image...]\n");
	for(int i = 0;i<timeout_ms;++i){
		usleep(1000);
		if(_imageAvailable[lDeviceId])
			break;
	}
	
	if(_imageAvailable[lDeviceId] != 1){
		printf(" [@(LJXA_ACQ_Acquire) timeout]\n");
		
		//Stop HighSpeed
		errCode = LJX8IF_StopHighSpeedDataCommunication(lDeviceId);
		printf("[@(LJXA_ACQ_Acquire) Stop HighSpeed](0x%x)\n",errCode);
		
		//Free memory
		if(_dataBuf[lDeviceId] != NULL){
			free(_dataBuf[lDeviceId]);
            _dataBuf[lDeviceId] = NULL;
		}
		return LJX8IF_RC_ERR_TIMEOUT;
	}
	printf(" [@(LJXA_ACQ_Acquire) done]\n");
	
	//Stop HighSpeed
	errCode = LJX8IF_StopHighSpeedDataCommunication(lDeviceId);
	printf("[@(LJXA_ACQ_Acquire) Stop HighSpeed](0x%x)\n",errCode);

//---------------------------------------------------------------------
//  Organize parameters related to the acquired image 
//---------------------------------------------------------------------
	unsigned short ZUnit;
	LJX8IF_GetZUnitSimpleArray(lDeviceId, &ZUnit);
	
	_getParam[lDeviceId].luminance_enabled 	= profileInfo.byLuminanceOutput;
	_getParam[lDeviceId].x_pointnum			= profileInfo.wProfileDataCount;
	_getParam[lDeviceId].y_linenum_acquired	= _lastImageSizeHeight[lDeviceId] * setParam->interpolateLines;
	_getParam[lDeviceId].x_pitch_um			= profileInfo.lXPitch * 0.01f;
	_getParam[lDeviceId].y_pitch_um			= setParam->y_pitch_um;
	_getParam[lDeviceId].z_pitch_um			= ZUnit * 0.01f;
	
	*getParam = _getParam[lDeviceId];

//---------------------------------------------------------------------
//  Copy internal buffer to user buffer
//---------------------------------------------------------------------
	int *dwDataBuf = (int*) &_dataBuf[lDeviceId][0];	
	int xDataNum = profileInfo.wProfileDataCount;
	int dataInterval = (profileInfo.byLuminanceOutput > 0) ? xDataNum*2 : xDataNum;
	
	for(int y=0; y < yDataNum; ++y){	
		for(int dy=0; dy < setParam->interpolateLines; ++dy){
		for(int x=0; x< xDataNum; ++x){
			int tmpHeightValue = dwDataBuf[6 + y*(dataInterval + 7) + x];
			
			int ny = y * setParam->interpolateLines + dy;
			
			//Convert height data. (Signed 32bit data to unsigned 16bit data.)
			//Irregular data is converted to zero.
			if(tmpHeightValue <= -2147483645){
					heightImage[ ny * xDataNum + x] = 0;
			}
			else{
					heightImage[ ny * xDataNum + x] = (unsigned short)(tmpHeightValue / ZUnit + 32768);
				}
			}
		}
	}
	
	if (profileInfo.byLuminanceOutput >0){
		for(int y=0; y < yDataNum; ++y){	
			for(int dy=0; dy < setParam->interpolateLines; ++dy){
			for(int x=0; x< xDataNum; ++x){
					
					int ny = y * setParam->interpolateLines + dy;
					
				//Convert luminance data. (Unsigned 32bit data to unsigned 8bit data.)
				//The range of valid value is 10bits, so 2-bit shift to bring it into 8bits data.
					luminanceImage[ ny * xDataNum + x] = (unsigned char)(dwDataBuf[6+ xDataNum + y *(dataInterval + 7) + x]>>2);
				}
			}
		}
	}
	
	
	//Free memory
	if(_dataBuf[lDeviceId] != NULL){
		free(_dataBuf[lDeviceId]);
        _dataBuf[lDeviceId] = NULL;
	}
	
	return LJX8IF_RC_OK;
}

int LJXA_ACQ_StartAsync(int lDeviceId, LJXA_ACQ_SETPARAM *setParam){
	int errCode;
	
	int yDataNum = setParam->y_linenum;
	//int timeout_ms = setParam->timeout_ms;
	int use_external_batchStart = setParam->use_external_batchStart;
	
	//Initialize
	errCode = LJX8IF_InitializeHighSpeedDataCommunication(lDeviceId, &_ethernetConfig[lDeviceId], _highSpeedPortNo[lDeviceId], &myCallbackFunc, yDataNum, lDeviceId);
	printf("[@(LJXA_ACQ_StartAsync) Initialize HighSpeed](0x%x)\n",errCode);
	if (errCode != LJX8IF_RC_OK) return errCode;
	
	//PreStart
	LJX8IF_HIGH_SPEED_PRE_START_REQ startReq;
	startReq.bySendPosition = 2;
	LJX8IF_PROFILE_INFO profileInfo;
	
	errCode = LJX8IF_PreStartHighSpeedDataCommunication(lDeviceId, &startReq, &profileInfo);
	printf("[@(LJXA_ACQ_StartAsync) PreStart](0x%x)\n",errCode);
	if (errCode != LJX8IF_RC_OK) return errCode;
	
	//Allocate memory
	if (_dataBuf[lDeviceId] != NULL) {
		free(_dataBuf[lDeviceId]);
		_dataBuf[lDeviceId] = NULL;
	}
	_dataBuf[lDeviceId] = (char*)malloc(yDataNum * (MAX_LJXA_XDATANUM * 2 + 7) * sizeof(int));
	if (_dataBuf[lDeviceId] == NULL) {
		return LJX8IF_RC_ERR_NOMEMORY;
	}

	//Start HighSpeed
	_imageAvailable[lDeviceId] = 0;
	_lastImageSizeHeight[ lDeviceId ] = 0;
	
	errCode = LJX8IF_StartHighSpeedDataCommunication(lDeviceId);
	printf("[@(LJXA_ACQ_StartAsync) Start HighSpeed](0x%x)\n",errCode);
	if (errCode != LJX8IF_RC_OK) {
		//Free memory
		if (_dataBuf[lDeviceId] != NULL) {
			free(_dataBuf[lDeviceId]);
			_dataBuf[lDeviceId] = NULL;
		}
		return errCode;
	}
	
	//StartMeasure(Batch Start)
	if ( use_external_batchStart > 0 ){
	}
	else{
		errCode = LJX8IF_StartMeasure(lDeviceId);
		printf("[@(LJXA_ACQ_StartAsync) Measure Start(Batch Start)](0x%x)\n",errCode);
	}
	
//---------------------------------------------------------------------
//  Organize parameters related to the acquired image
//---------------------------------------------------------------------
	unsigned short ZUnit;
	LJX8IF_GetZUnitSimpleArray(lDeviceId, &ZUnit);
	
	_getParam[lDeviceId].luminance_enabled 	= profileInfo.byLuminanceOutput;
	_getParam[lDeviceId].x_pointnum			= profileInfo.wProfileDataCount;
	//_getParam[lDeviceId].y_linenum_acquired : This parameter is unkown at this stage. Set later in "LJXA_ACQ_AcquireAsync" function.
	_getParam[lDeviceId].x_pitch_um			= profileInfo.lXPitch * 0.01f;
	_getParam[lDeviceId].y_pitch_um			= setParam->y_pitch_um;
	_getParam[lDeviceId].z_pitch_um			= ZUnit * 0.01f;
	
	return LJX8IF_RC_OK;
}

int LJXA_ACQ_AcquireAsync(int lDeviceId, unsigned short *heightImage, unsigned char *luminanceImage, LJXA_ACQ_SETPARAM *setParam, LJXA_ACQ_GETPARAM *getParam){
	int errCode;
	
	int yDataNum = setParam->y_linenum;
	//int timeout_ms = setParam->timeout_ms;
	//int use_external_batchStart = setParam->use_external_batchStart;
	
	//Allocated memory?
	if(_dataBuf[lDeviceId] == NULL){
		return LJX8IF_RC_ERR_NOMEMORY;
	}
	
	if(_imageAvailable[lDeviceId] != 1){
		//printf(" [No image has been acquired]\n");
		return LJX8IF_RC_ERR_TIMEOUT;
	}
	printf(" [@(LJXA_ACQ_AcquireAsync) done]\n");
		
	//Stop HighSpeed
	errCode = LJX8IF_StopHighSpeedDataCommunication(lDeviceId);
	printf("[@(LJXA_ACQ_AcquireAsync) Stop HighSpeed](0x%x)\n",errCode);

//---------------------------------------------------------------------
//  Organize parameters related to the acquired image
//---------------------------------------------------------------------
	//The rest parameters are preset in "LJXA_ACQ_StartAsync" function.
	_getParam[lDeviceId].y_linenum_acquired = _lastImageSizeHeight[lDeviceId] * setParam->interpolateLines;
	*getParam = _getParam[lDeviceId];
	
//---------------------------------------------------------------------
//  Copy internal buffer to user buffer
//---------------------------------------------------------------------
	int *dwDataBuf = (int*) &_dataBuf[lDeviceId][0];	
	int xDataNum = _getParam[lDeviceId].x_pointnum;
	int dataInterval = (_getParam[lDeviceId].luminance_enabled > 0) ? xDataNum*2 : xDataNum;
	
	unsigned short ZUnit;
	LJX8IF_GetZUnitSimpleArray(lDeviceId, &ZUnit);
	
	for(int y=0; y < yDataNum; ++y){
		for (int dy = 0; dy < setParam->interpolateLines; ++dy) {
		for (int x = 0; x < xDataNum; ++x) {
				int tmpHeightValue = dwDataBuf[6 + y * (dataInterval + 7) + x];

				int ny = y * setParam->interpolateLines + dy;

				//Convert height data. (Signed 32bit data to unsigned 16bit data.)
				//Irregular data is converted to zero.
				if (tmpHeightValue <= -2147483645) {
					heightImage[ny * xDataNum + x] = 0;
				}
				else {
					heightImage[ny * xDataNum + x] = (unsigned short)(tmpHeightValue / ZUnit + 32768);
				}
			}
		}
	}
	
	if (_getParam[lDeviceId].luminance_enabled >0){
		for(int y=0; y < yDataNum; ++y){	
			for(int dy=0; dy < setParam->interpolateLines; ++dy){
			for (int x = 0; x < xDataNum; ++x) {

					int ny = y * setParam->interpolateLines + dy;

				//Convert luminance data. (Unsigned 32bit data to unsigned 8bit data.)
				//The range of valid value is 10bits, so 2-bit shift to bring it into 8bits data.
					luminanceImage[ny * xDataNum + x] = (unsigned char)(dwDataBuf[6 + xDataNum + y * (dataInterval + 7) + x] >> 2);
				}
			}
		}
	}
	
	
	//Free memory
	if(_dataBuf[lDeviceId] != NULL){
		free(_dataBuf[lDeviceId]);
        _dataBuf[lDeviceId] = NULL;
	}
	return LJX8IF_RC_OK;
}


static void myCallbackFunc( unsigned char *Buf, unsigned int dwSize ,unsigned int dwCount, unsigned int dwNotify, unsigned int  dwUser){
	printf(" [@Callback func] dwSize:%d dwCount:%d notify:%d\n",(int)dwSize,(int)dwCount,(int)dwNotify);

	if((dwNotify == 0) || (dwNotify & 0x10000)){
		if ( dwCount != 0 ){
			if(_imageAvailable[dwUser] != 1){
				memcpy(&_dataBuf[dwUser][0],	Buf, dwSize*dwCount);
			
				_imageAvailable[dwUser] = 1;
				_lastImageSizeHeight[dwUser] = dwCount;
			}
		}
	}
}

