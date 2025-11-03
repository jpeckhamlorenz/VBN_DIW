# -*- coding: utf-8 -*-

import LJXAwrap
import ctypes
import sys
import time
import csv
import numpy
import PIL
import matplotlib.pyplot as plt


class LJXImageAcquisition:
    def __init__(self,
                 device_id=0,
                 ip=(192, 168, 0, 1),
                 command_port=24691,
                 highspeed_port=24692,
                 ysize=50,
                 timeout_sec=50,
                 use_external_batchStart=False):
        self.device_id = device_id
        self.ysize = ysize
        self.timeout_sec = timeout_sec
        self.use_external_batchStart = use_external_batchStart

        # Ethernet configuration
        self.ethernetConfig = LJXAwrap.LJX8IF_ETHERNET_CONFIG()
        for i, val in enumerate(ip):
            self.ethernetConfig.abyIpAddress[i] = val
        self.ethernetConfig.wPortNo = command_port
        self.highspeed_port = highspeed_port

        # Buffers and state
        self.xsize = None
        self.z_val = []
        self.lumi_val = []
        self.ysize_acquired = 0
        self.image_available = False
        self.profinfo = LJXAwrap.LJX8IF_PROFILE_INFO()

        # Bind callback
        self.my_callback = LJXAwrap.LJX8IF_CALLBACK_SIMPLE_ARRAY(self.callback)

    def callback(self, p_header, p_height, p_lumi,
                 luminance_enable, xpointnum, profnum, notify, user):
        """Called asynchronously when profiles are received."""
        if (notify == 0) or (notify == 0x10000):
            if profnum != 0 and not self.image_available:
                for i in range(xpointnum * profnum):
                    self.z_val[i] = p_height[i]
                    if luminance_enable == 1:
                        self.lumi_val[i] = p_lumi[i]
                self.ysize_acquired = profnum
                self.image_available = True

    def run(self):
        """Main acquisition process."""
        # Ethernet open
        res = LJXAwrap.LJX8IF_EthernetOpen(self.device_id, self.ethernetConfig)
        print("EthernetOpen:", hex(res))
        if res != 0:
            sys.exit("Failed to connect controller")

        # Init high-speed communication
        res = LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray(
            self.device_id,
            self.ethernetConfig,
            self.highspeed_port,
            self.my_callback,
            self.ysize,
            0)
        print("InitHighSpeedDataCommunicationSimpleArray:", hex(res))
        if res != 0:
            sys.exit("Init HS communication failed")

        # Pre-start
        req = LJXAwrap.LJX8IF_HIGH_SPEED_PRE_START_REQ()
        req.bySendPosition = 2
        res = LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication(
            self.device_id, req, self.profinfo)
        print("PreStartHighSpeedDataCommunication:", hex(res))
        if res != 0:
            sys.exit("PreStart failed")

        # Allocate buffers
        self.xsize = self.profinfo.wProfileDataCount
        self.z_val = [0] * self.xsize * self.ysize
        self.lumi_val = [0] * self.xsize * self.ysize

        # Start high-speed comm
        self.image_available = False
        res = LJXAwrap.LJX8IF_StartHighSpeedDataCommunication(self.device_id)
        print("StartHighSpeedDataCommunication:", hex(res))
        if res != 0:
            sys.exit("Start HS comm failed")

        # Start measure
        if not self.use_external_batchStart:
            LJXAwrap.LJX8IF_StartMeasure(self.device_id)

        # Wait for acquisition
        start_time = time.time()
        while True:
            if self.image_available:
                break
            if time.time() - start_time > self.timeout_sec:
                break

        # Stop
        LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(self.device_id)
        LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(self.device_id)
        LJXAwrap.LJX8IF_CommunicationClose(self.device_id)

        if not self.image_available:
            sys.exit("Failed to acquire image (timeout)")

        # Z unit scaling info
        ZUnit = ctypes.c_ushort()
        LJXAwrap.LJX8IF_GetZUnitSimpleArray(self.device_id, ZUnit)

        print("----------------------------------------")
        print(" Luminance output      :", self.profinfo.byLuminanceOutput)
        print(" Number of X points    :", self.profinfo.wProfileDataCount)
        print(" Number of Y lines     :", self.ysize_acquired)
        print(" X pitch in µm         :", self.profinfo.lXPitch / 100.0)
        print(" Z pitch in µm         :", ZUnit.value / 100.0)
        print("----------------------------------------")

        # Save to CSV
        self.save_to_csv("scan_profiles.csv")
        print("Saved", self.ysize_acquired, "profiles to scan_profiles.csv")

        # Plot example
        self.plot_example(ZUnit)

    def save_to_csv(self, filename):
        """Write z_val buffer to CSV."""
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(self.ysize_acquired):
                row = self.z_val[i * self.xsize:(i + 1) * self.xsize]
                writer.writerow(row)

    def plot_example(self, ZUnit):
        """Optional quick plots."""
        fig = plt.figure(figsize=(4.0, 6.0))
        plt.subplots_adjust(hspace=0.5)

        # Height image
        ax1 = fig.add_subplot(3, 1, 1)
        img1 = PIL.Image.new('I', (self.xsize, self.ysize))
        img1.putdata(list(map(int, self.z_val)))
        im_list1 = numpy.asarray(img1)
        ax1.imshow(im_list1, cmap='gray', vmin=0, vmax=65535,
                   interpolation='none')
        plt.title("Height Image")

        # Height profile
        ax3 = fig.add_subplot(3, 1, 3)
        sl = int(self.xsize * self.ysize_acquired / 2)
        x_val_mm = [0.0] * self.xsize
        z_val_mm = [0.0] * self.xsize
        for i in range(self.xsize):
            x_val_mm[i] = (self.profinfo.lXStart +
                           self.profinfo.lXPitch * i) / 1000.0
            if self.z_val[sl + i] == 0:
                z_val_mm[i] = numpy.nan
            else:
                z_val_mm[i] = int(self.z_val[sl + i]) - 32768
                z_val_mm[i] *= ZUnit.value / 100.0
                z_val_mm[i] /= 1000.0
        ax3.plot(x_val_mm, z_val_mm)
        plt.title("Height Profile")
        plt.show()


if __name__ == "__main__":
    acq = LJXImageAcquisition(
        ysize=1000,
        timeout_sec=50,
        use_external_batchStart=False
    )
    acq.run()