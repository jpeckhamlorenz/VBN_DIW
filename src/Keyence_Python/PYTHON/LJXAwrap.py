# -*- coding: 'unicode' -*-
# Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.
import ctypes
from ctypes import cdll
import os.path


#########################################################
# Select the library according to the Operating System
#########################################################
dll_name = "LJX8_IF.dll"        # For Windows
# dll_name = "libljxacom.so"    # For Linux

dllabspath = os.path.dirname(os.path.abspath(__file__))+os.path.sep+dll_name
mdll = cdll.LoadLibrary(dllabspath)


#########################################################
# Structure
#########################################################
class LJX8IF_ETHERNET_CONFIG(ctypes.Structure):
    _fields_ = [
        ("abyIpAddress", ctypes.c_ubyte * 4),
        ("wPortNo", ctypes.c_ushort),
        ("reserve", ctypes.c_ubyte * 2)]


class LJX8IF_TARGET_SETTING(ctypes.Structure):
    _fields_ = [
        ("byType", ctypes.c_ubyte),
        ("byCategory", ctypes.c_ubyte),
        ("byItem", ctypes.c_ubyte),
        ("reserve", ctypes.c_ubyte),
        ("byTarget1", ctypes.c_ubyte),
        ("byTarget2", ctypes.c_ubyte),
        ("byTarget3", ctypes.c_ubyte),
        ("byTarget4", ctypes.c_ubyte)]


class LJX8IF_PROFILE_INFO(ctypes.Structure):
    _fields_ = [
        ("byProfileCount", ctypes.c_ubyte),
        ("reserve1", ctypes.c_ubyte),
        ("byLuminanceOutput", ctypes.c_ubyte),
        ("reserve2", ctypes.c_ubyte),
        ("wProfileDataCount", ctypes.c_ushort),
        ("reserve3", ctypes.c_ubyte * 2),
        ("lXStart", ctypes.c_int),
        ("lXPitch", ctypes.c_int)]


class LJX8IF_PROFILE_HEADER(ctypes.Structure):
    _fields_ = [
        ("reserve", ctypes.c_uint),
        ("dwTriggerCount", ctypes.c_uint),
        ("lEncoderCount", ctypes.c_int),
        ("reserve2", ctypes.c_uint * 3)]


class LJX8IF_PROFILE_FOOTER(ctypes.Structure):
    _fields_ = [
        ("reserve", ctypes.c_uint)]


class LJX8IF_GET_PROFILE_REQUEST(ctypes.Structure):
    _fields_ = [
        ("byTargetBank", ctypes.c_ubyte),
        ("byPositionMode", ctypes.c_ubyte),
        ("reserve", ctypes.c_ubyte * 2),
        ("dwGetProfileNo", ctypes.c_uint),
        ("byGetProfileCount", ctypes.c_ubyte),
        ("byErase", ctypes.c_ubyte),
        ("reserve2", ctypes.c_ubyte * 2)]


class LJX8IF_GET_BATCH_PROFILE_REQUEST(ctypes.Structure):
    _fields_ = [
        ("byTargetBank", ctypes.c_ubyte),
        ("byPositionMode", ctypes.c_ubyte),
        ("reserve", ctypes.c_ubyte * 2),
        ("dwGetBatchNo", ctypes.c_uint),
        ("dwGetProfileNo", ctypes.c_uint),
        ("byGetProfileCount", ctypes.c_ubyte),
        ("byErase", ctypes.c_ubyte),
        ("reserve2", ctypes.c_ubyte * 2)]


class LJX8IF_GET_PROFILE_RESPONSE(ctypes.Structure):
    _fields_ = [
        ("dwCurrentProfileNo", ctypes.c_uint),
        ("dwOldestProfileNo", ctypes.c_uint),
        ("dwGetTopProfileNo", ctypes.c_uint),
        ("byGetProfileCount", ctypes.c_ubyte),
        ("reserve", ctypes.c_ubyte * 3)]


class LJX8IF_GET_BATCH_PROFILE_RESPONSE(ctypes.Structure):
    _fields_ = [
        ("dwCurrentBatchNo", ctypes.c_uint),
        ("dwCurrentBatchProfileCount", ctypes.c_uint),
        ("dwOldestBatchNo", ctypes.c_uint),
        ("dwOldestBatchProfileCount", ctypes.c_uint),
        ("dwGetBatchNo", ctypes.c_uint),
        ("dwGetBatchProfileCount", ctypes.c_uint),
        ("dwGetBatchTopProfileNo", ctypes.c_uint),
        ("byGetProfileCount", ctypes.c_ubyte),
        ("byCurrentBatchCommited", ctypes.c_ubyte),
        ("reserve", ctypes.c_ubyte * 2)]


class LJX8IF_HIGH_SPEED_PRE_START_REQ(ctypes.Structure):
    _fields_ = [
        ("bySendPosition", ctypes.c_ubyte),
        ("reserve", ctypes.c_ubyte * 3)]


#########################################################
# DLL Wrapper Function
#########################################################

LJX8IF_CALLBACK_SIMPLE_ARRAY = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.POINTER(LJX8IF_PROFILE_HEADER),  # pProfileHeaderArray
    ctypes.POINTER(ctypes.c_ushort),        # pHeightProfileArray
    ctypes.POINTER(ctypes.c_ushort),        # pLuminanceProfileArray
    ctypes.c_uint,                          # dwLuminanceEnable
    ctypes.c_uint,                          # dwProfileDataCount
    ctypes.c_uint,                          # dwCount
    ctypes.c_uint,                          # dwNotify
    ctypes.c_uint                           # dwUser
    )

LJX8IF_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_void_p,                        # pBuffer
    ctypes.c_uint,			    # dwSize
    ctypes.c_uint,	                    # dwCount
    ctypes.c_uint,	                    # dwNotify
    ctypes.c_uint	                    # dwUser
    )

# LJX8IF_EthernetOpen
LJX8IF_EthernetOpen = mdll.LJX8IF_EthernetOpen
LJX8IF_EthernetOpen.restype = ctypes.c_int
LJX8IF_EthernetOpen.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(LJX8IF_ETHERNET_CONFIG)  # pEthernetConfig
    ]

# LJX8IF_CommunicationClose
LJX8IF_CommunicationClose = mdll.LJX8IF_CommunicationClose
LJX8IF_CommunicationClose.restype = ctypes.c_int
LJX8IF_CommunicationClose.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_RebootController
LJX8IF_RebootController = mdll.LJX8IF_RebootController
LJX8IF_RebootController.restype = ctypes.c_int
LJX8IF_RebootController.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_ReturnToFactorySetting
LJX8IF_ReturnToFactorySetting = mdll.LJX8IF_ReturnToFactorySetting
LJX8IF_ReturnToFactorySetting.restype = ctypes.c_int
LJX8IF_ReturnToFactorySetting.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_ControlLaser
LJX8IF_ControlLaser = mdll.LJX8IF_ControlLaser
LJX8IF_ControlLaser.restype = ctypes.c_int
LJX8IF_ControlLaser.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_ubyte                          # byState
    ]

# LJX8IF_GetError
LJX8IF_GetError = mdll.LJX8IF_GetError
LJX8IF_GetError.restype = ctypes.c_int
LJX8IF_GetError.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_ubyte,                         # byReceivedMax
    ctypes.POINTER(ctypes.c_ubyte),         # pbyErrCount
    ctypes.POINTER(ctypes.c_ushort)         # pwErrCode
    ]

# LJX8IF_ClearError
LJX8IF_ClearError = mdll.LJX8IF_ClearError
LJX8IF_ClearError.restype = ctypes.c_int
LJX8IF_ClearError.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_ushort                         # wErrCode
    ]

# LJX8IF_TrgErrorReset
LJX8IF_TrgErrorReset = mdll.LJX8IF_TrgErrorReset
LJX8IF_TrgErrorReset.restype = ctypes.c_int
LJX8IF_TrgErrorReset.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_GetTriggerAndPulseCount
LJX8IF_GetTriggerAndPulseCount = mdll.LJX8IF_GetTriggerAndPulseCount
LJX8IF_GetTriggerAndPulseCount.restype = ctypes.c_int
LJX8IF_GetTriggerAndPulseCount.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(ctypes.c_uint),          # pdwTriggerCount
    ctypes.POINTER(ctypes.c_int)            # plEncoderCount
    ]

# LJX8IF_SetTimerCount
LJX8IF_SetTimerCount = mdll.LJX8IF_SetTimerCount
LJX8IF_SetTimerCount.restype = ctypes.c_int
LJX8IF_SetTimerCount.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_uint                           # dwTimerCount
    ]

# LJX8IF_GetTimerCount
LJX8IF_GetTimerCount = mdll.LJX8IF_GetTimerCount
LJX8IF_GetTimerCount.restype = ctypes.c_int
LJX8IF_GetTimerCount.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(ctypes.c_uint)           # pdwTimerCount
    ]

# LJX8IF_GetHeadTemperature
LJX8IF_GetHeadTemperature = mdll.LJX8IF_GetHeadTemperature
LJX8IF_GetHeadTemperature.restype = ctypes.c_int
LJX8IF_GetHeadTemperature.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(ctypes.c_short),         # pnSensorTemperature
    ctypes.POINTER(ctypes.c_short),         # pnProcessorTemperature
    ctypes.POINTER(ctypes.c_short)          # pnCaseTemperature
    ]

# LJX8IF_GetHeadModel
LJX8IF_GetHeadModel = mdll.LJX8IF_GetHeadModel
LJX8IF_GetHeadModel.restype = ctypes.c_int
LJX8IF_GetHeadModel.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_char_p                         # pHeadModel
    ]

# LJX8IF_GetSerialNumber
LJX8IF_GetSerialNumber = mdll.LJX8IF_GetSerialNumber
LJX8IF_GetSerialNumber.restype = ctypes.c_int
LJX8IF_GetSerialNumber.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_char_p,                        # pControllerSerialNo
    ctypes.c_char_p                         # pHeadSerialNo
    ]

# LJX8IF_GetAttentionStatus
LJX8IF_GetAttentionStatus = mdll.LJX8IF_GetAttentionStatus
LJX8IF_GetAttentionStatus.restype = ctypes.c_int
LJX8IF_GetAttentionStatus.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(ctypes.c_ushort)         # pwAttentionStatus
    ]

# LJX8IF_Trigger
LJX8IF_Trigger = mdll.LJX8IF_Trigger
LJX8IF_Trigger.restype = ctypes.c_int
LJX8IF_Trigger.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_StartMeasure
LJX8IF_StartMeasure = mdll.LJX8IF_StartMeasure
LJX8IF_StartMeasure.restype = ctypes.c_int
LJX8IF_StartMeasure.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_StopMeasure
LJX8IF_StopMeasure = mdll.LJX8IF_StopMeasure
LJX8IF_StopMeasure.restype = ctypes.c_int
LJX8IF_StopMeasure.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_ClearMemory
LJX8IF_ClearMemory = mdll.LJX8IF_ClearMemory
LJX8IF_ClearMemory.restype = ctypes.c_int
LJX8IF_ClearMemory.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_SetSetting
LJX8IF_SetSetting = mdll.LJX8IF_SetSetting
LJX8IF_SetSetting.restype = ctypes.c_int
LJX8IF_SetSetting.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_ubyte,                         # byDepth
    LJX8IF_TARGET_SETTING,                  # TargetSetting
    ctypes.POINTER(ctypes.c_ubyte),         # pData
    ctypes.c_uint,                          # dwDataSize
    ctypes.POINTER(ctypes.c_uint)           # pdwError
    ]

# LJX8IF_GetSetting
LJX8IF_GetSetting = mdll.LJX8IF_GetSetting
LJX8IF_GetSetting.restype = ctypes.c_int
LJX8IF_GetSetting.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_ubyte,                         # byDepth
    LJX8IF_TARGET_SETTING,                  # TargetSetting
    ctypes.POINTER(ctypes.c_ubyte),         # pData
    ctypes.c_uint                           # dwDataSize
    ]

# LJX8IF_InitializeSetting
LJX8IF_InitializeSetting = mdll.LJX8IF_InitializeSetting
LJX8IF_InitializeSetting.restype = ctypes.c_int
LJX8IF_InitializeSetting.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_ubyte,                         # byDepth
    ctypes.c_ubyte                          # byTarget
    ]

# LJX8IF_ReflectSetting
LJX8IF_ReflectSetting = mdll.LJX8IF_ReflectSetting
LJX8IF_ReflectSetting.restype = ctypes.c_int
LJX8IF_ReflectSetting.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_ubyte,                         # byDepth
    ctypes.POINTER(ctypes.c_uint)           # pdwError
    ]

# LJX8IF_RewriteTemporarySetting
LJX8IF_RewriteTemporarySetting = mdll.LJX8IF_RewriteTemporarySetting
LJX8IF_RewriteTemporarySetting.restype = ctypes.c_int
LJX8IF_RewriteTemporarySetting.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_ubyte                          # byDepth
    ]

# LJX8IF_CheckMemoryAccess
LJX8IF_CheckMemoryAccess = mdll.LJX8IF_CheckMemoryAccess
LJX8IF_CheckMemoryAccess.restype = ctypes.c_int
LJX8IF_CheckMemoryAccess.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(ctypes.c_ubyte)          # pbyBusy
    ]

# LJX8IF_SetXpitch
LJX8IF_SetXpitch = mdll.LJX8IF_SetXpitch
LJX8IF_SetXpitch.restype = ctypes.c_int
LJX8IF_SetXpitch.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_uint                           # dwXpitch
    ]

# LJX8IF_GetXpitch
LJX8IF_GetXpitch = mdll.LJX8IF_GetXpitch
LJX8IF_GetXpitch.restype = ctypes.c_int
LJX8IF_GetXpitch.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(ctypes.c_uint)           # pdwXpitch
    ]

# LJX8IF_ChangeActiveProgram
LJX8IF_ChangeActiveProgram = mdll.LJX8IF_ChangeActiveProgram
LJX8IF_ChangeActiveProgram.restype = ctypes.c_int
LJX8IF_ChangeActiveProgram.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.c_ubyte                          # byProgramNo
    ]

# LJX8IF_GetActiveProgram
LJX8IF_GetActiveProgram = mdll.LJX8IF_GetActiveProgram
LJX8IF_GetActiveProgram.restype = ctypes.c_int
LJX8IF_GetActiveProgram.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(ctypes.c_ubyte)          # pbyProgramNo
    ]

# LJX8IF_GetProfile
LJX8IF_GetProfile = mdll.LJX8IF_GetProfile
LJX8IF_GetProfile.restype = ctypes.c_int
LJX8IF_GetProfile.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(LJX8IF_GET_PROFILE_REQUEST),         # pReq
    ctypes.POINTER(LJX8IF_GET_PROFILE_RESPONSE),        # pRsp
    ctypes.POINTER(LJX8IF_PROFILE_INFO),    # pProfileInfo
    ctypes.POINTER(ctypes.c_int),           # pdwProfileData
    ctypes.c_uint                           # dwDataSize
    ]

# LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray
LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray = mdll.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray
LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray.restype = ctypes.c_int
LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray.argtypes = [
    ctypes.c_int,                               # lDeviceId
    ctypes.POINTER(LJX8IF_ETHERNET_CONFIG),     # pEthernetConfig
    ctypes.c_ushort,                            # wHighSpeedPortNo
    LJX8IF_CALLBACK_SIMPLE_ARRAY,               # pCallBackSimpleArray
    ctypes.c_uint,                              # dwProfileCount
    ctypes.c_uint                               # dwThreadId
    ]

# LJX8IF_PreStartHighSpeedDataCommunication
LJX8IF_PreStartHighSpeedDataCommunication = mdll.LJX8IF_PreStartHighSpeedDataCommunication
LJX8IF_PreStartHighSpeedDataCommunication.restype = ctypes.c_int
LJX8IF_PreStartHighSpeedDataCommunication.argtypes = [
    ctypes.c_int,                                       # lDeviceId
    ctypes.POINTER(LJX8IF_HIGH_SPEED_PRE_START_REQ),    # pReq
    ctypes.POINTER(LJX8IF_PROFILE_INFO)                 # pProfileInfo
    ]

# LJX8IF_StartHighSpeedDataCommunication
LJX8IF_StartHighSpeedDataCommunication = mdll.LJX8IF_StartHighSpeedDataCommunication
LJX8IF_StartHighSpeedDataCommunication.restype = ctypes.c_int
LJX8IF_StartHighSpeedDataCommunication.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_StopHighSpeedDataCommunication
LJX8IF_StopHighSpeedDataCommunication = mdll.LJX8IF_StopHighSpeedDataCommunication
LJX8IF_StopHighSpeedDataCommunication.restype = ctypes.c_int
LJX8IF_StopHighSpeedDataCommunication.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_FinalizeHighSpeedDataCommunication
LJX8IF_FinalizeHighSpeedDataCommunication = mdll.LJX8IF_FinalizeHighSpeedDataCommunication
LJX8IF_FinalizeHighSpeedDataCommunication.restype = ctypes.c_int
LJX8IF_FinalizeHighSpeedDataCommunication.argtypes = [
    ctypes.c_int                            # lDeviceId
    ]

# LJX8IF_GetZUnitSimpleArray
LJX8IF_GetZUnitSimpleArray = mdll.LJX8IF_GetZUnitSimpleArray
LJX8IF_GetZUnitSimpleArray.restype = ctypes.c_int
LJX8IF_GetZUnitSimpleArray.argtypes = [
    ctypes.c_int,                           # lDeviceId
    ctypes.POINTER(ctypes.c_ushort)         # pwZUnit
    ]
