//Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.
/** @file
@brief	LJX8_IF_linux Header
*/

#ifndef _LJX8_IF_LINUX_H
#define _LJX8_IF_LINUX_H

#define MAX_LJXA_DEVICENUM	(6U)
#define MAX_LJXA_XDATANUM	(3200U)
#define MAX_LJXA_YDATANUM	(42000U)

/// Setting value storage level designation
typedef enum {
	LJX8IF_SETTING_DEPTH_WRITE		= 0x00,		// Write settings area
	LJX8IF_SETTING_DEPTH_RUNNING	= 0x01,		// Running settings area
	LJX8IF_SETTING_DEPTH_SAVE		= 0x02,		// Save area
} LJX8IF_SETTING_DEPTH;

/// Initialization target setting item designation
typedef enum {
	LJX8IF_INIT_SETTING_TARGET_PRG0		= 0x00,		// Program 0
	LJX8IF_INIT_SETTING_TARGET_PRG1		= 0x01,		// Program 1
	LJX8IF_INIT_SETTING_TARGET_PRG2		= 0x02,		// Program 2
	LJX8IF_INIT_SETTING_TARGET_PRG3		= 0x03,		// Program 3
	LJX8IF_INIT_SETTING_TARGET_PRG4		= 0x04,		// Program 4
	LJX8IF_INIT_SETTING_TARGET_PRG5		= 0x05,		// Program 5
	LJX8IF_INIT_SETTING_TARGET_PRG6		= 0x06,		// Program 6
	LJX8IF_INIT_SETTING_TARGET_PRG7		= 0x07,		// Program 7
	LJX8IF_INIT_SETTING_TARGET_PRG8		= 0x08,		// Program 8
	LJX8IF_INIT_SETTING_TARGET_PRG9		= 0x09,		// Program 9
	LJX8IF_INIT_SETTING_TARGET_PRG10	= 0x0A,		// Program 10
	LJX8IF_INIT_SETTING_TARGET_PRG11	= 0x0B,		// Program 11
	LJX8IF_INIT_SETTING_TARGET_PRG12	= 0x0C,		// Program 12
	LJX8IF_INIT_SETTING_TARGET_PRG13	= 0x0D,		// Program 13
	LJX8IF_INIT_SETTING_TARGET_PRG14	= 0x0E,		// Program 14
	LJX8IF_INIT_SETTING_TARGET_PRG15	= 0x0F,		// Program 15
} LJX8IF_INIT_SETTING_TARGET;

/// Get profile target buffer designation
typedef enum {
	LJX8IF_PROFILE_BANK_ACTIVE		= 0x00,		// Active surface
	LJX8IF_PROFILE_BANK_INACTIVE	= 0x01,		// Inactive surface
} LJX8IF_PROFILE_BANK;

/// Get profile position specification method designation (batch measurement: off)
typedef enum {
	LJX8IF_PROFILE_POSITION_CURRENT	= 0x00,		// From current
	LJX8IF_PROFILE_POSITION_OLDEST	= 0x01,		// From oldest
	LJX8IF_PROFILE_POSITION_SPEC	= 0x02,		// Specify position
} LJX8IF_PROFILE_POSITION;

/// Get profile batch data position specification method designation (batch measurement: on)
typedef enum {
	LJX8IF_BATCH_POSITION_CURRENT		= 0x00,		// From current
	LJX8IF_BATCH_POSITION_SPEC			= 0x02,		// Specify position
	LJX8IF_BATCH_POSITION_COMMITED		= 0x03,		// From current after batch commitment
	LJX8IF_BATCH_POSITION_CURRENT_ONLY	= 0x04,		// Current only
} LJX8IF_BATCH_POSITION;

/// Version info structure
typedef struct {
	int	nMajorNumber;		// Major number
	int	nMinorNumber;		// Minor number
	int	nRevisionNumber;	// Revision number
	int	nBuildNumber;		// Buiid number
} LJX8IF_VERSION_INFO;

/// Ethernet settings structure
typedef struct {
	unsigned char	abyIpAddress[4];	// The IP address of the controller to connect to.
	unsigned short	wPortNo;			// The port number of the controller to connect to.
	unsigned char	reserve[2];			// Reserved
} LJX8IF_ETHERNET_CONFIG;

/// Setting item designation structure
typedef struct {
	unsigned char	byType;			// Setting type
	unsigned char	byCategory;		// Category
	unsigned char	byItem;			// Setting item
	unsigned char	reserve;		// Reserved
	unsigned char	byTarget1;		// Setting Target 1
	unsigned char	byTarget2;		// Setting Target 2
	unsigned char	byTarget3;		// Setting Target 3
	unsigned char	byTarget4;		// Setting Target 4
} LJX8IF_TARGET_SETTING;

/// Profile information
typedef struct {
	unsigned char	byProfileCount;		// Fixed to 1.(reserved for future use)
	unsigned char	reserve1;			// Reserved
	unsigned char	byLuminanceOutput;	// Whether luminance output is on.
	unsigned char	reserve2;			// Reserved
	unsigned short	wProfileDataCount;	// Profile data count
	unsigned char	reserve3[2];		// Reserved
	int				lXStart;			// 1st point X coordinate.
	int				lXPitch;			// Profile data X direction interval.
} LJX8IF_PROFILE_INFO;

/// Profile header information structure
typedef struct {
	unsigned int	reserve;		// Reserved
	unsigned int	dwTriggerCount;	// The trigger count when the trigger was issued.
	int				lEncoderCount;	// The encoder count when the trigger was issued.
	unsigned int	reserve2[3];	// Reserved
} LJX8IF_PROFILE_HEADER;

/// Profile footer information structure
typedef struct {
	unsigned int	reserve;	// Reserved
} LJX8IF_PROFILE_FOOTER;

/// Get profile request structure (batch measurement: off)
typedef struct {
	unsigned char	byTargetBank;		// The target surface to read.
	unsigned char	byPositionMode;		// The get profile position specification method.
	unsigned char	reserve[2];			// Reserved
	unsigned int	dwGetProfileNo;		// The profile number for the profile to get.
	unsigned char	byGetProfileCount;	// The number of profiles to read.
	unsigned char	byErase;			// Specifies whether to erase the profile data that was read and the profile data older than that.
	unsigned char	reserve2[2];		// Reserved
} LJX8IF_GET_PROFILE_REQUEST;

/// Get profile request structure (batch measurement: on)
typedef struct {
	unsigned char	byTargetBank;		// The target surface to read.
	unsigned char	byPositionMode;		// The get profile position specification method
	unsigned char	reserve[2];			// Reserved
	unsigned int	dwGetBatchNo;		// The batch number for the profile to get
	unsigned int	dwGetProfileNo;		// The profile number to start getting profiles from in the specified batch number.
	unsigned char	byGetProfileCount;	// The number of profiles to read.
	unsigned char	byErase;			// Specifies whether to erase the profile data that was read and the profile data older than that.
	unsigned char	reserve2[2];		// Reserved
} LJX8IF_GET_BATCH_PROFILE_REQUEST;

/// Get profile response structure (batch measurement: off)
typedef struct {
	unsigned int	dwCurrentProfileNo;		// The profile number at the current point in time.
	unsigned int	dwOldestProfileNo;		// The profile number for the oldest profile held by the controller.
	unsigned int	dwGetTopProfileNo;		// The profile number for the oldest profile out of those that were read this time.
	unsigned char	byGetProfileCount;		// The number of profiles that were read this time.	
	unsigned char	reserve[3];				// Reserved
} LJX8IF_GET_PROFILE_RESPONSE;

/// Get profile response structure (batch measurement: on)
typedef struct {
	unsigned int	dwCurrentBatchNo;			// The batch number at the current point in time.
	unsigned int	dwCurrentBatchProfileCount;	// The number of profiles in the newest batch.
	unsigned int	dwOldestBatchNo;			// The batch number for the oldest batch held by the controller.
	unsigned int	dwOldestBatchProfileCount;	// The number of profiles in the oldest batch held by the controller.
	unsigned int	dwGetBatchNo;				// The batch number that was read this time.
	unsigned int	dwGetBatchProfileCount;		// The number of profiles in the batch that was read this time.
	unsigned int	dwGetBatchTopProfileNo;		// The oldest profile number in the batch out of the profiles that were read this time.
	unsigned char	byGetProfileCount;			// The number of profiles that were read this time.
	unsigned char	byCurrentBatchCommited;		// The batch measurements for the newest batch number has finished.
	unsigned char	reserve[2];					// Reserved
} LJX8IF_GET_BATCH_PROFILE_RESPONSE;

/// High-speed communication prep start request structure
typedef struct {
	unsigned char	bySendPosition;			// Send start position
	unsigned char	reserve[3];				// Reserved
} LJX8IF_HIGH_SPEED_PRE_START_REQ;


/**
Callback function interface for high-speed data communication
@param	pBuffer		A pointer to the buffer that stores the profile data.
@param	dwSize		The size in BYTEs per single unit of the profile.
@param	dwCount		The number of profiles stored in pBuffer.
@param	dwNotify	Notification of an interruption in high-speed communication or a break in batch measurements.
@param	dwUser		User information
*/
typedef void(*LJX8IF_CALLBACK)(unsigned char* pBuffer, unsigned int dwSize, unsigned int dwCount, unsigned int dwNotify, unsigned int dwUser);

/**
Callback function interface for high-speed data communication
@param	pProfileHeaderArray		A pointer to the buffer that stores the header data array.
@param	pHeightProfileArray		A pointer to the buffer that stores the profile data array.
@param	pLuminanceProfileArray		A pointer to the buffer that stores the luminance profile data array.
@param	dwLuminanceEnable		The value indicating whether luminance data output is enable or not.
@param	dwProfileDataCount		The data count of one profile.
@param	dwCount		The number of profile or header data stored in buffer.
@param	dwNotify	Notification of an interruption in high-speed communication or a break in batch measurements.
@param	dwUser		User information
*/
typedef void(*LJX8IF_CALLBACK_SIMPLE_ARRAY)(LJX8IF_PROFILE_HEADER* pProfileHeaderArray, unsigned short* pHeightProfileArray, unsigned short* pLuminanceProfileArray, unsigned int dwLuminanceEnable, unsigned int dwProfileDataCount, unsigned int dwCount, unsigned int dwNotify, unsigned int dwUser);



extern "C"
{
	/**
	Ethernet communication connection
	@param	lDeviceId		The communication device to communicate with.
	@param	pEthernetConfig	Ethernet communication settings
	@return	Return code
	*/
	int LJX8IF_EthernetOpen(int lDeviceId, LJX8IF_ETHERNET_CONFIG* pEthernetConfig);
	
	/**
	Disconnect communication path
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_CommunicationClose(int lDeviceId);

	// System control
	/**
	Reboot the controller
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_RebootController(int lDeviceId);

	/**
	Return to factory state
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_ReturnToFactorySetting(int lDeviceId);

	/**
	Control Laser
	@param	lDeviceId	The communication device to communicate with.
	@param	byState		Laser state
	@return	Return code
	*/
	int LJX8IF_ControlLaser(int lDeviceId, unsigned char byState);

	/**
	Get system error information
	@param	lDeviceId		The communication device to communicate with.
	@param	byReceivedMax	The maximum amount of system error information to receive
	@param	pbyErrCount		The buffer to receive the amount of system error information.
	@param	pwErrCode		The buffer to receive the system error information.
	@return	Return code
	*/
	int LJX8IF_GetError(int lDeviceId, unsigned char byReceivedMax, unsigned char* pbyErrCount, unsigned short* pwErrCode);

	/**
	Clear system error
	@param	lDeviceId	The communication device to communicate with.
	@param	wErrCode	The error code for the error you wish to clear.
	@return	Return code
	*/
	int LJX8IF_ClearError(int lDeviceId, unsigned short wErrCode);

	/**
	Reset TRG_ERROR
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_TrgErrorReset(int lDeviceId);

	/**
	Get trigger and encoder pulse count
	@param	lDeviceId		The communication device to communicate with.
	@param	pdwTriggerCount	The buffer to receive trigger count
	@param	plEncoderCount	The buffer to receive encoder pulse count
	@return	Return code
	*/
	int LJX8IF_GetTriggerAndPulseCount(int lDeviceId, unsigned int* pdwTriggerCount, int* plEncoderCount);

	/**
	Set Timer count
	@param	lDeviceId		The communication device to communicate with.
	@param	dwTimerCount	Timer count after the change.
	@return	Return code
	*/
	int LJX8IF_SetTimerCount(int lDeviceId, unsigned int dwTimerCount);

	/**
	Get Timer count
	@param	lDeviceId		The communication device to communicate with.
	@param	pdwTimerCount	The buffer to receive Timer count.
	@return	Return code
	*/
	int LJX8IF_GetTimerCount(int lDeviceId, unsigned int* pdwTimerCount);

	/**
	Get head temperature
	@param	lDeviceId				The communication device to communicate with.
	@param	pnSensorTemperature		The buffer to receive sensor Temperature.
	@param	pnProcessorTemperature	The buffer to receive processor Temperature.
	@param	pnCaseTemperature		The buffer to receive case Temperature.
	@return	Return code
	*/
	int LJX8IF_GetHeadTemperature(int lDeviceId, short* pnSensorTemperature, short* pnProcessorTemperature, short* pnCaseTemperature);

	/**
	Get head model
	@param	lDeviceId			The communication device to communicate with.
	@param	pHeadModel			The buffer to receive head model
	@return	Return code
	*/
	int LJX8IF_GetHeadModel(int lDeviceId, char* pHeadModel);

	/**
	Get serial Number
	@param	lDeviceId			The communication device to communicate with.
	@param	pControllerSerialNo	The buffer to receive serial number of the controller
	@param	pHeadSerialNo		The buffer to receive serial number of the head
	@return	Return code
	*/
	int LJX8IF_GetSerialNumber(int lDeviceId, char* pControllerSerialNo, char* pHeadSerialNo);
	
	/**
	Get current attention status value
	@param	lDeviceId		The communication device to communicate with.
	@param	pwAttentionStatus	The buffer to receive attention status
	@return	Return code
	*/
	int LJX8IF_GetAttentionStatus(int lDeviceId, unsigned short* pwAttentionStatus);

	// Measurement control
	/**
	Trigger
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_Trigger(int lDeviceId);
	
	/**
	Start batch measurements
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_StartMeasure(int lDeviceId);

	/**
	Stop batch measurements
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_StopMeasure(int lDeviceId);

	/**
	Clear memory
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_ClearMemory(int lDeviceId);

	// Functions related to modifying or reading settings
	/**
	Send setting
	@param	lDeviceId		The communication device to communicate with.
	@param	byDepth			The level to reflect the setting value
	@param	TargetSetting	The item that is the target
	@param	pData			The buffer that stores the setting data
	@param	dwDataSize		The size in BYTEs of the setting data
	@param	pdwError		Detailed setting error
	@return	Return code
	*/
	int LJX8IF_SetSetting(int lDeviceId, unsigned char byDepth, LJX8IF_TARGET_SETTING TargetSetting, void* pData, unsigned int dwDataSize, unsigned int* pdwError);

	/**
	Get setting
	@param	lDeviceId		The communication device to communicate with.
	@param	byDepth			The level of the setting value to get.
	@param	TargetSetting	The item that is the target
	@param	pData			The buffer to receive the setting data
	@param	dwDataSize		The size of the buffer to receive the acquired data in BYTEs.
	@return	Return code
	*/
	int LJX8IF_GetSetting(int lDeviceId, unsigned char byDepth, LJX8IF_TARGET_SETTING TargetSetting, void* pData, unsigned int dwDataSize);

	/**
	Initialize setting
	@param	lDeviceId	The communication device to communicate with.
	@param	byDepth		The level to reflect the initialized setting.
	@param	byTarget	The setting that is the target for initialization.
	@return	Return code
	*/
	int LJX8IF_InitializeSetting(int lDeviceId, unsigned char byDepth, unsigned char byTarget);

	/**
	Request to reflect settings in the write settings area
	@param	lDeviceId	The communication device to communicate with.
	@param	byDepth		The level to reflect the setting value
	@param	pdwError	Detailed setting error
	@return	Return code
	*/
	int LJX8IF_ReflectSetting(int lDeviceId, unsigned char byDepth, unsigned int* pdwError);

	/**
	Update write settings area
	@param	lDeviceId	The communication device to communicate with.
	@param	byDepth		The level of the settings to update the write settings area with.
	@return	Return code
	*/
	int LJX8IF_RewriteTemporarySetting(int lDeviceId, unsigned char byDepth);

	/**
	Check the status of saving to the save area
	@param	lDeviceId	The communication device to communicate with.
	@param	pbyBusy		Other than 0: Accessing the save area, 0: no access.
	@return	Return code
	*/
	int LJX8IF_CheckMemoryAccess(int lDeviceId, unsigned char* pbyBusy);


	/**
	Set Xpitch
	@param	lDeviceId	The communication device to communicate with.
	@param	dwXpitch	Xpitch after the change.
	@return	Return code
	*/
	int LJX8IF_SetXpitch(int lDeviceId, unsigned int dwXpitch);

	/**
	Get Xpitch
	@param	lDeviceId		The communication device to communicate with.
	@param	pdwXpitch		The buffer to receive Xpitch.
	@return	Return code
	*/
	int LJX8IF_GetXpitch(int lDeviceId, unsigned int* pdwXpitch);

	/**
	Change program
	@param	lDeviceId	The communication device to communicate with.
	@param	byProgramNo	Program number after the change.
	@return	Return code
	*/
	int LJX8IF_ChangeActiveProgram(int lDeviceId, unsigned char byProgramNo);

	/**
	Get the active program number
	@param	lDeviceId		The communication device to communicate with.
	@param	pbyProgramNo	The buffer to receive the active program number.
	@return	Return code
	*/
	int LJX8IF_GetActiveProgram(int lDeviceId, unsigned char* pbyProgramNo);

	// Acquiring measurement results
	/**
	Get profiles
	@param	lDeviceId		The communication device to communicate with.
	@param	pReq			The position, etc., of the profiles to get.
	@param	pRsp			The position, etc., of the profiles that were actually acquired.
	@param	pProfileInfo	The profile information for the acquired profiles.
	@param	pdwProfileData	The buffer to get the profile data.
	@param	dwDataSize		pdwProfileData size in BYTEs
	@return	Return code
	*/
	int LJX8IF_GetProfile(int lDeviceId, LJX8IF_GET_PROFILE_REQUEST* pReq, LJX8IF_GET_PROFILE_RESPONSE* pRsp, LJX8IF_PROFILE_INFO* pProfileInfo, unsigned int* pdwProfileData, unsigned int dwDataSize);
	/**
	Get batch profiles
	@param	lDeviceId		The communication device to communicate with.
	@param	pReq			The position, etc., of the profiles to get.
	@param	pRsp			The position, etc., of the profiles that were actually acquired.
	@param	pProfileInfo	The profile information for the acquired profiles.
	@param	pdwBatchData	The buffer to get the profile data.
	@param	dwDataSize		pdwProfileData size in BYTEs
	@return	Return code
	*/
	int LJX8IF_GetBatchProfile(int lDeviceId, LJX8IF_GET_BATCH_PROFILE_REQUEST* pReq, LJX8IF_GET_BATCH_PROFILE_RESPONSE* pRsp, LJX8IF_PROFILE_INFO * pProfileInfo, unsigned int* pdwBatchData, unsigned int dwDataSize);

	/**
	Get batch profiles by simple array format
	@param	lDeviceId				The communication device to communicate with.
	@param	pReq					The position, etc., of the profiles to get.
	@param	pRsp					The position, etc., of the profiles that were actually acquired.
	@param	pProfileInfo			The profile information for the acquired profiles.
	@param  pProfileHeaderArray		The buffer to get array of header.
	@param  pHeightProfileArray		The buffer to get array of profile data.
	@param  pLuminanceProfileArray	The buffer to get array of luminance profile data.
	@return	Return code
	*/
	int LJX8IF_GetBatchSimpleArray(int lDeviceId, LJX8IF_GET_BATCH_PROFILE_REQUEST* pReq, LJX8IF_GET_BATCH_PROFILE_RESPONSE* pRsp, LJX8IF_PROFILE_INFO* pProfileInfo, LJX8IF_PROFILE_HEADER* pProfileHeaderArray, unsigned short* pHeightProfileArray, unsigned short* pLuminanceProfileArray);

	// High-speed data communication related
	/**
	Initialize Ethernet high-speed data communication
	@param	lDeviceId			The communication device to communicate with.
	@param	pEthernetConfig		The Ethernet settings used in high-speed communication.
	@param	wHighSpeedPortNo	The port number used in high-speed communication.
	@param	pCallBack			The callback function to call when data is received by high-speed communication.
	@param	dwProfileCount		The frequency to call the callback function.
	@param	dwThreadId			Thread ID.
	@return	Return code
	*/
	int LJX8IF_InitializeHighSpeedDataCommunication(int lDeviceId, LJX8IF_ETHERNET_CONFIG* pEthernetConfig, unsigned short wHighSpeedPortNo,
		LJX8IF_CALLBACK pCallBack, unsigned int dwProfileCount, unsigned int dwThreadId);

	/**
	Initialize Ethernet high-speed data communication for simple array
	@param	lDeviceId				The communication device to communicate with.
	@param	pEthernetConfig			The Ethernet settings used in high-speed communication.
	@param	wHighSpeedPortNo		The port number used in high-speed communication.
	@param	pCallBackSimpleArray	The callback function to call when data is received by high-speed communication.
	@param	dwProfileCount			The frequency to call the callback function.
	@param	dwThreadId				Thread ID.
	@return	Return code
	*/
	int LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray(int lDeviceId, LJX8IF_ETHERNET_CONFIG* pEthernetConfig, unsigned short wHighSpeedPortNo,
		LJX8IF_CALLBACK_SIMPLE_ARRAY pCallBackSimpleArray, unsigned int dwProfileCount, unsigned int dwThreadId);

	/**
	Request preparation before starting high-speed data communication
	@param	lDeviceId		The communication device to communicate with.
	@param	pReq			What data to send high-speed communication from.
	@param	pProfileInfo	Stores the profile information.
	@return	Return code
	*/
	int LJX8IF_PreStartHighSpeedDataCommunication(int lDeviceId, LJX8IF_HIGH_SPEED_PRE_START_REQ* pReq, LJX8IF_PROFILE_INFO* pProfileInfo);

	/**
	Start high-speed data communication
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_StartHighSpeedDataCommunication(int lDeviceId);

	/**
	Stop high-speed data communication
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_StopHighSpeedDataCommunication(int lDeviceId);

	/**
	Finalize high-speed data communication
	@param	lDeviceId	The communication device to communicate with.
	@return	Return code
	*/
	int LJX8IF_FinalizeHighSpeedDataCommunication(int lDeviceId);

	/**
	Get the simpleArray conversion factor
	@param	lDeviceId	The communication device to communicate with.
	@param	pwZUnit		The buffer to receive the simpleArray conversion factor.
	@return	Return code
	*/
	int LJX8IF_GetZUnitSimpleArray(int lDeviceId, unsigned short* pwZUnit);

};
#endif /* _LJX8_IF_LINUX_H */