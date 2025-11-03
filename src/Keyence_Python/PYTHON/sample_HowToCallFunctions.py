# -*- coding: 'unicode' -*-
# Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.

import LJXAwrap

import ctypes
import sys
import time


def main():

    deviceId = 0  # Set "0" if you use only 1 head.
    ethernetConfig = LJXAwrap.LJX8IF_ETHERNET_CONFIG()
    ethernetConfig.abyIpAddress[0] = 192  # IP address
    ethernetConfig.abyIpAddress[1] = 168
    ethernetConfig.abyIpAddress[2] = 0
    ethernetConfig.abyIpAddress[3] = 1
    ethernetConfig.wPortNo = 24691        # Port No.

    res = LJXAwrap.LJX8IF_EthernetOpen(deviceId, ethernetConfig)
    print("LJXAwrap.LJX8IF_EthernetOpen:", hex(res))
    if res != 0:
        print("Failed to connect contoller.")
        sys.exit()
    print("----")

    ##################################################################
    # sample_HowToCallFunctions.py:
    #  A sample collection of how to call LJXAwrap I/F functions.
    #
    # Conditional branch of each sample is initially set to 'False'.
    # This is to prevent accidental execution. Set 'True' to execute.
    #
    # <NOTE> Controller settings may change in some sample codes.
    #
    ##################################################################

    if False:
        print("LJXAwrap.LJX8IF_RebootController:",
              hex(LJXAwrap.LJX8IF_RebootController(deviceId)))
        print("Rebooting contoller.")
        print("----")
        print("\nTerminated normally.")
        sys.exit()

    if False:
        print("LJXAwrap.LJX8IF_ReturnToFactorySetting:",
              hex(LJXAwrap.LJX8IF_ReturnToFactorySetting(deviceId)))
        print("----")

    if False:
        print("LJXAwrap.LJX8IF_ControlLaser:",
              hex(LJXAwrap.LJX8IF_ControlLaser(deviceId, 0)), "<Laser OFF>")
        time.sleep(1)
        print("LJXAwrap.LJX8IF_ControlLaser:",
              hex(LJXAwrap.LJX8IF_ControlLaser(deviceId, 1)), "<Laser ON>")
        print("----")

    if False:
        rcvMax = 10
        errCnt = ctypes.c_ubyte()
        errCode = (ctypes.c_ushort * rcvMax)()
        res = LJXAwrap.LJX8IF_GetError(deviceId, rcvMax, errCnt, errCode)
        print("LJXAwrap.LJX8IF_GetError:", hex(res),
              "<rcvMax, errCount, errCode>=",
              rcvMax, errCnt.value, hex(errCode[0]))

        res = LJXAwrap.LJX8IF_ClearError(deviceId, errCode[0])
        print("LJXAwrap.LJX8IF_ClearError:", hex(res),
              "<clear error code>=", errCode[0])
        print("----")

    if False:
        print("LJXAwrap.LJX8IF_TrgErrorReset:",
              hex(LJXAwrap.LJX8IF_TrgErrorReset(deviceId)))
        print("----")

    if False:
        trgCnt = ctypes.c_uint()
        encCnt = ctypes.c_int()
        res = LJXAwrap.LJX8IF_GetTriggerAndPulseCount(deviceId, trgCnt, encCnt)
        print("LJXAwrap.LJX8IF_GetTriggerAndPulseCount:", hex(res),
              "<TriggerCnt, EncoderCnt>=", trgCnt.value, encCnt.value)
        print("----")

    if False:
        timerCount_set = 0
        print("LJXAwrap.LJX8IF_SetTimerCount:",
              hex(LJXAwrap.LJX8IF_SetTimerCount(deviceId, timerCount_set)),
              "<TimerCount_set>=", timerCount_set)

        timerCount_get = ctypes.c_uint()
        res = LJXAwrap.LJX8IF_GetTimerCount(deviceId, timerCount_get)
        print("LJXAwrap.LJX8IF_GetTimerCount:", hex(res),
              "<TimerCount [sec]>=", timerCount_get.value/10000.0)
        time.sleep(1)

        timerCount_get = ctypes.c_uint()
        res = LJXAwrap.LJX8IF_GetTimerCount(deviceId, timerCount_get)
        print("LJXAwrap.LJX8IF_GetTimerCount:", hex(res),
              "<TimerCount [sec]>=", timerCount_get.value/10000.0)
        time.sleep(1)

        timerCount_get = ctypes.c_uint()
        res = LJXAwrap.LJX8IF_GetTimerCount(deviceId, timerCount_get)
        print("LJXAwrap.LJX8IF_GetTimerCount:", hex(res),
              "<TimerCount [sec]>=", timerCount_get.value/10000.0)
        print("----")

    if False:
        sensorT = ctypes.c_short()
        processorT = ctypes.c_short()
        caseT = ctypes.c_short()
        res = LJXAwrap.LJX8IF_GetHeadTemperature(deviceId,
                                                 sensorT, processorT, caseT)
        print("LJXAwrap.LJX8IF_GetHeadTemperature:", hex(res),
              "<SensorT, ProcessorT, CaseT [degree Celsius]>=",
              sensorT.value/100.0, processorT.value/100.0, caseT.value/100.0)
        print("----")

    if False:
        headmodel = ctypes.create_string_buffer(32)
        res = LJXAwrap.LJX8IF_GetHeadModel(deviceId, headmodel)
        print("LJXAwrap.LJX8IF_GetHeadModel:", hex(res),
              "<headmodel>=", headmodel.value)
        print("----")

    if False:
        controllerSerial = ctypes.create_string_buffer(16)
        headSerial = ctypes.create_string_buffer(16)
        res = LJXAwrap.LJX8IF_GetSerialNumber(deviceId,
                                              controllerSerial, headSerial)
        print("LJXAwrap.LJX8IF_GetSerialNumber:", hex(res),
              "<controllerSerial>=", controllerSerial.value,
              "<headSerial>=", headSerial.value)
        print("----")

    if False:
        attentionStatus = ctypes.c_ushort()
        res = LJXAwrap.LJX8IF_GetAttentionStatus(deviceId, attentionStatus)
        print("LJXAwrap.LJX8IF_GetAttentionStatus:", hex(res),
              "<AttentionStatus>=", bin(attentionStatus.value))
        print("----")

    if False:
        print("LJXAwrap.LJX8IF_Trigger:",
              hex(LJXAwrap.LJX8IF_Trigger(deviceId)))
        print("----")

    if False:
        print("LJXAwrap.LJX8IF_StartMeasure:",
              hex(LJXAwrap.LJX8IF_StartMeasure(deviceId)))
        time.sleep(1)
        print("LJXAwrap.LJX8IF_StopMeasure:",
              hex(LJXAwrap.LJX8IF_StopMeasure(deviceId)))
        print("----")

    if False:
        print("LJXAwrap.LJX8IF_ClearMemory:",
              hex(LJXAwrap.LJX8IF_ClearMemory(deviceId)))
        print("----")

    if True:
        # Example of how to change some settings.
        # In this example, the "sampling cycle" of Program No,0 is changed.

        depth = 1  # 0: Write, 1: Running, 2: Save

        targetSetting = LJXAwrap.LJX8IF_TARGET_SETTING()
        targetSetting.byType = 0x10         # Program No.0
        targetSetting.byCategory = 0x00     # Trigger Category
        targetSetting.byItem = 0x02         # Sampling Cycle
        targetSetting.byTarget1 = 0x00      # reserved
        targetSetting.byTarget2 = 0x00      # reserved
        targetSetting.byTarget3 = 0x00      # reserved
        targetSetting.byTarget4 = 0x00      # reserved

        dataSize = 4

        # Set the sampling cylce to '100Hz'
        err = ctypes.c_uint()
        pyArr = [3, 0, 0, 0]  # Sampling Cycle setting value. 3: 100Hz
        settingData_set = (ctypes.c_ubyte * dataSize)(*pyArr)

        res = LJXAwrap.LJX8IF_SetSetting(deviceId, depth,
                                         targetSetting,
                                         settingData_set, dataSize, err)
        print("LJXAwrap.LJX8IF_SetSetting:", hex(res),
              "<Set value>=", settingData_set[0],
              "<SettingError>=", hex(err.value))

        # Get setting. This is not mandatory. Just to confirm.
        settingData_get = (ctypes.c_ubyte * dataSize)()
        res = LJXAwrap.LJX8IF_GetSetting(deviceId, depth,
                                         targetSetting,
                                         settingData_get, dataSize)
        print("LJXAwrap.LJX8IF_GetSetting:", hex(res),
              "<Get value>=", settingData_get[0])

        # Set the sampling cylce to '1kHz'
        pyArr = [6, 0, 0, 0]  # Sampling Cycle setting value. 6: 1kHz
        settingData_set = (ctypes.c_ubyte * dataSize)(*pyArr)
        res = LJXAwrap.LJX8IF_SetSetting(deviceId, depth,
                                         targetSetting,
                                         settingData_set, dataSize, err)
        print("LJXAwrap.LJX8IF_SetSetting:", hex(res),
              "<Set value>=", settingData_set[0],
              "<SettingError>=", hex(err.value))

        # Get setting. This is not mandatory. Just to confirm.
        res = LJXAwrap.LJX8IF_GetSetting(deviceId, depth,
                                         targetSetting,
                                         settingData_get, dataSize)
        print("LJXAwrap.LJX8IF_GetSetting:", hex(res),
              "<Get value>=", settingData_get[0])
        print("----")

    if False:
        # Example of how to initialize settings.
        # In this example, whole settings of Program No.0 will be initialized.

        Depth = 1   # 0: Write, 1: Running, 2: Save
        Target = 0  # Program No.0
        res = LJXAwrap.LJX8IF_InitializeSetting(deviceId, Depth, Target)
        print("LJXAwrap.LJX8IF_InitializeSetting:", hex(res),
              "<Initialize Program No.>=", Target)
        print("----")

    if False:
        # Example of how to rewrite and reflect Settings.
        # In this example,
        # the currently running settings are overwritten
        # with the contents of the non-volatile memory.

        # (1)Copy settings from "Save area" to "Write area"
        Depth = 2  # 1: Running, 2: Save
        Error = ctypes.c_uint()
        res = LJXAwrap.LJX8IF_RewriteTemporarySetting(deviceId, Depth)
        print("LJXAwrap.LJX8IF_RewriteTemporarySetting:", hex(res))

        # (2)Reflect settings from "Write area" to "Running area"
        Depth = 1  # 1: Running, 2: Save
        res = LJXAwrap.LJX8IF_ReflectSetting(deviceId, Depth, Error)
        print("LJXAwrap.LJX8IF_ReflectSetting:", hex(res),
              "<SettingError>=", hex(Error.value))
        print("----")

    if False:
        Busy = ctypes.c_ubyte()
        res = LJXAwrap.LJX8IF_CheckMemoryAccess(deviceId, Busy)
        print("LJXAwrap.LJX8IF_CheckMemoryAccess:", hex(res),
              "<Busy>=", Busy.value)
        print("----")

    if False:
        # Example of how to change active Program No.

        # Set active program No. to '5'
        programNo_set = 5
        res = LJXAwrap.LJX8IF_ChangeActiveProgram(deviceId, programNo_set)
        print("LJXAwrap.LJX8IF_ChangeActiveProgram:", hex(res),
              "<ProgramNo_set>=", programNo_set)

        # Get active program No.
        programNo_get = ctypes.c_ubyte()
        res = LJXAwrap.LJX8IF_GetActiveProgram(deviceId, programNo_get)
        print("LJXAwrap.LJX8IF_GetActiveProgram:", hex(res),
              "<ProgramNo_get>=", programNo_get.value)

        time.sleep(1)

        # Set active program No. to '0'
        programNo_set = 0
        res = LJXAwrap.LJX8IF_ChangeActiveProgram(deviceId, programNo_set)
        print("LJXAwrap.LJX8IF_ChangeActiveProgram:", hex(res),
              "<ProgramNo_set>=", programNo_set)

        # Get active program No.
        programNo_get = ctypes.c_ubyte()
        res = LJXAwrap.LJX8IF_GetActiveProgram(deviceId, programNo_get)
        print("LJXAwrap.LJX8IF_GetActiveProgram:", hex(res),
              "<ProgramNo_get>=", programNo_get.value)
        print("----")

    if False:
        xpitch_get = ctypes.c_uint()
        res = LJXAwrap.LJX8IF_GetXpitch(deviceId, xpitch_get)
        print("LJXAwrap.LJX8IF_GetXpitch:", hex(res),
              "<Xpitch_get>=", xpitch_get.value)

        xpitch_backup = xpitch_get.value

        xpitch_set = xpitch_get.value + 5
        res = LJXAwrap.LJX8IF_SetXpitch(deviceId, xpitch_set)
        print("LJXAwrap.LJX8IF_SetXpitch:", hex(res),
              "<Xpitch_set>=", xpitch_set)

        xpitch_get = ctypes.c_uint()
        res = LJXAwrap.LJX8IF_GetXpitch(deviceId, xpitch_get)
        print("LJXAwrap.LJX8IF_GetXpitch:", hex(res),
              "<Xpitch_get>=", xpitch_get.value)

        xpitch_set = xpitch_backup
        res = LJXAwrap.LJX8IF_SetXpitch(deviceId, xpitch_set)
        print("LJXAwrap.LJX8IF_SetXpitch:", hex(res),
              "<Xpitch_set>=", xpitch_set)

        xpitch_get = ctypes.c_uint()
        res = LJXAwrap.LJX8IF_GetXpitch(deviceId, xpitch_get)
        print("LJXAwrap.LJX8IF_GetXpitch:", hex(res),
              "<Xpitch_get>=", xpitch_get.value)
        print("----")

    if False:
        # Example of how to get profile data.
        #
        # <NOTE>
        # -This method is suitable for reading a few profile data.
        #
        # -Use high-speed communication method to acquire a large amount
        #  of profiles, such as height or luminance image data.
        #  For details, refer to another sample (sample_ImageAcquisition.py)

        # Change according to your controller settings.
        xpointNum = 800            # Number of X points per one profile.
        withLumi = 0              # 1: luminance data exists, 0: not exists.

        # Specifies the position, etc. of the profiles to get.
        req = LJXAwrap.LJX8IF_GET_PROFILE_REQUEST()
        req.byTargetBank = 0x0      # 0: active bank
        req.byPositionMode = 0x0    # 0: from current position
        req.dwGetProfileNo = 0x0    # use when position mode is "POSITION_SPEC"
        req.byGetProfileCount = 1   # the number of profiles to read.
        req.byErase = 0             # 0: Do not erase

        rsp = LJXAwrap.LJX8IF_GET_PROFILE_RESPONSE()

        profinfo = LJXAwrap.LJX8IF_PROFILE_INFO()

        # Calculate the buffer size to store the received profile data.
        dataSize = ctypes.sizeof(LJXAwrap.LJX8IF_PROFILE_HEADER)
        dataSize += ctypes.sizeof(LJXAwrap.LJX8IF_PROFILE_FOOTER)
        dataSize += ctypes.sizeof(ctypes.c_uint) * xpointNum * (1 + withLumi)
        dataSize *= req.byGetProfileCount

        dataNumIn4byte = int(dataSize / ctypes.sizeof(ctypes.c_uint))
        profdata = (ctypes.c_int * dataNumIn4byte)()

        # Send command.
        res = LJXAwrap.LJX8IF_GetProfile(deviceId,
                                         req,
                                         rsp,
                                         profinfo,
                                         profdata,
                                         dataSize)

        print("LJXAwrap.LJX8IF_GetProfile:", hex(res))
        if res != 0:
            print("Failed to get profile.")
            sys.exit()

        print("----------------------------------------")
        print(" byLuminanceOutput     :", profinfo.byLuminanceOutput)
        print(" wProfileDataCount(X)  :", profinfo.wProfileDataCount)
        print(" lXPitch(in 0.01um)    :", profinfo.lXPitch)
        print(" lXStart(in 0.01um)    :", profinfo.lXStart)
        print("-----")
        print(" dwCurrentProfileNo    :", rsp.dwCurrentProfileNo)
        print(" dwOldestProfileNo     :", rsp.dwOldestProfileNo)
        print(" dwGetTopProfileNo     :", rsp.dwGetTopProfileNo)
        print(" byGetProfileCount     :", rsp.byGetProfileCount)
        print("----------------------------------------")

        headerSize = ctypes.sizeof(LJXAwrap.LJX8IF_PROFILE_HEADER)
        addressOffset_height = int(headerSize / ctypes.sizeof(ctypes.c_uint))
        addressOffset_lumi = addressOffset_height + profinfo.wProfileDataCount

        for i in range(profinfo.wProfileDataCount):
            # Conver X data to the actual length in millimeters
            x_val_mm = (profinfo.lXStart + profinfo.lXPitch * i) / 100.0  # um
            x_val_mm /= 1000.0  # mm

            # Conver Z data to the actual length in millimeters
            z_val = profdata[addressOffset_height + i]

            if z_val <= -2147483645:  # invalid value
                z_val_mm = - 999.9999
            else:
                z_val_mm = z_val / 100.0  # um
                z_val_mm /= 1000.0  # mm

            # Luminance data
            lumi_val = profdata[addressOffset_lumi + i]

            print('{:.04f}'.format(x_val_mm),
                  '{:.04f}'.format(z_val_mm),
                  lumi_val)

        print("----")

    res = LJXAwrap.LJX8IF_CommunicationClose(deviceId)
    print("LJXAwrap.LJX8IF_CommunicationClose:", hex(res))

    return


if __name__ == '__main__':
    main()
