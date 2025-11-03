# -*- coding: utf-8 -*-

import LJXAwrap
import ctypes
import sys
import time
import csv
import numpy


class LJXContinuousAcquisition:
    def __init__(self,
                 device_id=0,
                 ip=(192, 168, 0, 1),
                 command_port=24691,
                 highspeed_port=24692,
                 timeout_sec=5,
                 use_external_batchStart=False,
                 csv_filename="scan_stream.csv"):

        self.device_id = device_id
        self.timeout_sec = timeout_sec
        self.use_external_batchStart = use_external_batchStart
        self.csv_filename = csv_filename

        # Ethernet config
        self.ethernetConfig = LJXAwrap.LJX8IF_ETHERNET_CONFIG()
        for i, val in enumerate(ip):
            self.ethernetConfig.abyIpAddress[i] = val
        self.ethernetConfig.wPortNo = command_port
        self.highspeed_port = highspeed_port

        # Buffers
        self.xsize = None
        self.profinfo = LJXAwrap.LJX8IF_PROFILE_INFO()

        # Bind callback
        self.my_callback = LJXAwrap.LJX8IF_CALLBACK_SIMPLE_ARRAY(self.callback)

        # Open CSV file immediately
        self.csv_file = open(self.csv_filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.profile_count = 0

    def callback(self, p_header, p_height, p_lumi,
                 luminance_enable, xpointnum, profnum, notify, user):
        """Called whenever new profiles arrive."""
        if (notify == 0) or (notify == 0x10000):
            if profnum > 0:
                # Write each profile row-by-row
                for row_i in range(profnum):
                    start = row_i * xpointnum
                    end = (row_i + 1) * xpointnum
                    row = [int(val) for val in p_height[start:end]]
                    self.csv_writer.writerow(row)
                    self.profile_count += 1

                if self.profile_count % 100 == 0:
                    print(f"Saved {self.profile_count} profiles...")

    def run(self):
        """Run until Ctrl+C."""
        # Open communication
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
            1000,   # 0 = unlimited until stop
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

        self.xsize = self.profinfo.wProfileDataCount
        print(f"Xsize = {self.xsize} points per profile")

        # Start high-speed comm
        res = LJXAwrap.LJX8IF_StartHighSpeedDataCommunication(self.device_id)
        print("StartHighSpeedDataCommunication:", hex(res))
        if res != 0:
            sys.exit("Start HS comm failed")

        # Start measure
        if not self.use_external_batchStart:
            LJXAwrap.LJX8IF_StartMeasure(self.device_id)

        # print("Acquisition started. Press Ctrl+C to stop...")

        try:
            while True:
                time.sleep(0.1)  # keep main thread alive
        except KeyboardInterrupt:
            print("\nStopping acquisition...")

        # Stop and cleanup
        LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(self.device_id)
        LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(self.device_id)
        LJXAwrap.LJX8IF_CommunicationClose(self.device_id)

        self.csv_file.close()
        print(f"Saved {self.profile_count} profiles to {self.csv_filename}")
        
        
        
        
        
        print("Acquisition started. Will stop if no data for",
              self.data_timeout, "seconds...")

        try:
            while True:
                time.sleep(0.2)
                if time.time() - self.last_received_time > self.data_timeout:
                    print("No data received for",
                          self.data_timeout, "seconds → stopping.")
                    break
        finally:
            # Cleanup
            LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(self.device_id)
            LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(self.device_id)
            LJXAwrap.LJX8IF_CommunicationClose(self.device_id)

            self.csv_file.close()
            print(f"Saved {self.profile_count} profiles to {self.csv_filename}")




if __name__ == "__main__":
    acq = LJXContinuousAcquisition(
        use_external_batchStart=False,
        csv_filename="scan_stream.csv"
    )
    acq.run()


























































