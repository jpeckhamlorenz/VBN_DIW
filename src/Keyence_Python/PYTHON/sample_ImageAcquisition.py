# -*- coding: 'unicode' -*-
# Copyright (c) 2021 KEYENCE CORPORATION. All rights reserved.

import LJXAwrap

import ctypes
import sys
import time

import numpy
import PIL
import matplotlib.pyplot as plt

##############################################################################
# sample_ImageAcquisition.py: LJ-X8000A Image acquisition sample.
#
# -First half part: Describes how to acquire images via LJXAwrap I/F.
# -Second half part: Describes how to display images using additional modules.
#
##############################################################################

image_available = False  # Flag to conrirm the completion of image acquisition.
ysize_acquired = 0       # Number of Y lines of acquired image.
z_val = []               # The buffer for height image.
lumi_val = []            # The buffer for luminance image.


def main():

    global image_available
    global ysize_acquired
    global z_val
    global lumi_val

    ##################################################################
    # CHANGE THIS BLOCK TO MATCH YOUR SENSOR SETTINGS (FROM HERE)
    ##################################################################

    deviceId = 0                        # Set "0" if you use only 1 head.
    ysize = 50                  # Number of Y lines.
    timeout_sec = 50                     # Timeout value for the acquiring image
    use_external_batchStart = False     # 'True' if you start batch externally.

    ethernetConfig = LJXAwrap.LJX8IF_ETHERNET_CONFIG()
    ethernetConfig.abyIpAddress[0] = 192    # IP address
    ethernetConfig.abyIpAddress[1] = 168
    ethernetConfig.abyIpAddress[2] = 0
    ethernetConfig.abyIpAddress[3] = 1
    ethernetConfig.wPortNo = 24691          # Port No.
    HighSpeedPortNo = 24692                 # Port No. for high-speed

    ##################################################################
    # CHANGE THIS BLOCK TO MATCH YOUR SENSOR SETTINGS (TO HERE)
    ##################################################################

    # Ethernet open
    res = LJXAwrap.LJX8IF_EthernetOpen(0, ethernetConfig)
    print("LJXAwrap.LJX8IF_EthernetOpen:", hex(res))
    if res != 0:
        print("Failed to connect contoller.")
        print("Exit the program.")
        sys.exit()

    # Initialize Hi-Speed Communication
    my_callback_s_a = LJXAwrap.LJX8IF_CALLBACK_SIMPLE_ARRAY(callback_s_a)

    res = LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray(
        deviceId,
        ethernetConfig,
        HighSpeedPortNo,
        my_callback_s_a,
        ysize,
        0)
    print("LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray:",
          hex(res))
    if res != 0:
        print("\nExit the program.")
        sys.exit()

    # PreStart Hi-Speed Communication
    req = LJXAwrap.LJX8IF_HIGH_SPEED_PRE_START_REQ()
    req.bySendPosition = 2
    profinfo = LJXAwrap.LJX8IF_PROFILE_INFO()

    res = LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication(
        deviceId,
        req,
        profinfo)
    print("LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication:", hex(res))
    if res != 0:
        print("\nExit the program.")
        sys.exit()

    # allocate the memory
    xsize = profinfo.wProfileDataCount
    z_val = [0] * xsize * ysize
    lumi_val = [0] * xsize * ysize

    # Start Hi-Speed Communication
    image_available = False
    res = LJXAwrap.LJX8IF_StartHighSpeedDataCommunication(deviceId)
    print("LJXAwrap.LJX8IF_StartHighSpeedDataCommunication:", hex(res))
    if res != 0:
        print("\nExit the program.")
        sys.exit()

    # Start Measure (Start Batch)
    if use_external_batchStart is False:
        LJXAwrap.LJX8IF_StartMeasure(deviceId)

    # wait for the image acquisition complete
    start_time = time.time()
    while True:
        if image_available:
            break
        if time.time() - start_time > timeout_sec:
            break

    # Stop
    res = LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(deviceId)
    print("LJXAwrap.LJX8IF_StoptHighSpeedDataCommunication:", hex(res))

    # Finalize
    res = LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(deviceId)
    print("LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication:", hex(res))

    # Close
    res = LJXAwrap.LJX8IF_CommunicationClose(deviceId)
    print("LJXAwrap.LJX8IF_CommunicationClose:", hex(res))

    if image_available is not True:
        print("\nFailed to acquire image (timeout)")
        print("\nTerminated normally.")
        sys.exit()
    
    
    # =====================================================
    # Save acquired height profiles to CSV
    # =====================================================
    import csv
    
    with open("scan_profiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(ysize_acquired):
            row = z_val[i * xsize : (i + 1) * xsize]
            writer.writerow(row)
    
    print("Saved", ysize_acquired, "profiles to scan_profiles.csv")
    
    ##################################################################
    # Information of the acquired image
    ##################################################################
    ZUnit = ctypes.c_ushort()
    LJXAwrap.LJX8IF_GetZUnitSimpleArray(deviceId, ZUnit)

    print("----------------------------------------")
    print(" Luminance output      : ", profinfo.byLuminanceOutput)
    print(" Number of X points    : ", profinfo.wProfileDataCount)
    print(" Number of Y lines     : ", ysize_acquired)
    print(" X pitch in micrometer : ", profinfo.lXPitch / 100.0)
    print(" Z pitch in micrometer : ", ZUnit.value / 100.0)
    print("----------------------------------------")

    ##################################################################
    # Display part:
    #
    # <NOTE> Additional modules are required to execute the next block.
    # -'Numpy' for handling array data.
    # -'Pillow' for 2D image display.
    # -'matplotlib' for profile display.
    #
    # If you want to skip,
    # set the next conditional branch to 'False'.
    #
    ##################################################################
    
    
    if True:
        fig = plt.figure(figsize=(4.0, 6.0))
        plt.subplots_adjust(hspace=0.5)

        # Height image display
        ax1 = fig.add_subplot(3, 1, 1)
        img1 = PIL.Image.new('I', (xsize, ysize))
        img1.putdata(list(map(int, z_val)))
        im_list1 = numpy.asarray(img1)

        ax1.imshow(im_list1,
                   cmap='gray',
                   vmin=0,
                   vmax=65535,
                   interpolation='none')

        plt.title("Height Image")

        # Luminance image display
        ax2 = fig.add_subplot(3, 1, 2)
        img2 = PIL.Image.new('I', (xsize, ysize))
        img2.putdata(list(map(int, lumi_val)))
        im_list2 = numpy.asarray(img2)

        ax2.imshow(im_list2,
                   cmap='gray',
                   vmin=0,
                   vmax=1024,
                   interpolation='none')

        plt.title("Luminance Image")

        # Height profile display
        ax3 = fig.add_subplot(3, 1, 3)
        sl = int(xsize * ysize_acquired / 2)  # the horizontal center profile

        x_val_mm = [0.0] * xsize
        z_val_mm = [0.0] * xsize
        for i in range(xsize):
            # Conver X data to the actual length in millimeters
            x_val_mm[i] = (profinfo.lXStart + profinfo.lXPitch * i)/100.0  # um
            x_val_mm[i] /= 1000.0  # mm

            # Conver Z data to the actual length in millimeters
            if z_val[sl + i] == 0:  # invalid value
                z_val_mm[i] = numpy.nan
            else:
                # 'Simple array data' is offset to be unsigned 16-bit data.
                # Decode by subtracting 32768 to get a signed value.
                z_val_mm[i] = int(z_val[sl + i]) - 32768  # decode
                z_val_mm[i] *= ZUnit.value / 100.0  # um
                z_val_mm[i] /= 1000.0  # mm

        plotz_min = numpy.nanmin(z_val_mm)
        if numpy.isnan(plotz_min):
            plotz_min = -1.0
        else:
            plotz_min -= 1.0

        plotz_max = numpy.nanmax(z_val_mm)
        if numpy.isnan(plotz_max):
            plotz_max = 1.0
        else:
            plotz_max += 1.0

        plt.ylim(plotz_min, plotz_max)

        ax3.plot(x_val_mm, z_val_mm)

        plt.title("Height Profile")

        # Show all plot
        print("\nPress 'q' key to exit the program...")
        plt.show()
        #plt.close('all')

    print("\nTerminated normally.")
    return


###############################################################################
# Callback function
# It is called when the specified number of profiles are received.
###############################################################################
def callback_s_a(p_header,
                 p_height,
                 p_lumi,
                 luminance_enable,
                 xpointnum,
                 profnum,
                 notify, user):

    global ysize_acquired
    global image_available
    global z_val
    global lumi_val

    if (notify == 0) or (notify == 0x10000):
        if profnum != 0:
            if image_available is False:
                for i in range(xpointnum * profnum):
                    z_val[i] = p_height[i]
                    if luminance_enable == 1:
                        lumi_val[i] = p_lumi[i]

                ysize_acquired = profnum
                image_available = True
    return


if __name__ == '__main__':
    main()
