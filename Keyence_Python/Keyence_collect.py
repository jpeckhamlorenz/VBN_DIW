# -*- coding: utf-8 -*-

import LJXAwrap
import ctypes
import sys
import time
import csv
import numpy


class LJXAutoStopAcquisition:
    def __init__(self,
                 device_id=0,
                 ip=(192, 168, 0, 1),
                 command_port=24691,
                 highspeed_port=24692,
                 data_timeout=10.0,   # seconds without data before stopping
                 use_external_batchStart=False,
                 csv_filename="scan_stream.csv",
                 auto_restart=True,
                 restart_delay=2.0,   # seconds to wait between cycles
                 max_cycles=None):    # None = unlimited cycles

        self.device_id = device_id
        self.data_timeout = data_timeout
        self.use_external_batchStart = use_external_batchStart
        self.csv_filename = csv_filename
        self.auto_restart = auto_restart
        self.restart_delay = restart_delay
        self.max_cycles = max_cycles

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

        # CSV file (will be opened/closed per cycle)
        self.csv_file = None
        self.csv_writer = None

        # Cycle tracking
        self.current_cycle = 0
        self.total_profiles_collected = 0
        self.running = True

    def callback(self, p_header, p_height, p_lumi,
                 luminance_enable, xpointnum, profnum, notify, user):
        """Called whenever new profiles arrive."""
        if (notify == 0) or (notify == 0x10000):
            if profnum > 0 and self.csv_writer is not None:
                # Write each profile row-by-row
                for row_i in range(profnum):
                    start = row_i * xpointnum
                    end = (row_i + 1) * xpointnum
                    row = [int(val) for val in p_height[start:end]]
                    self.csv_writer.writerow(row)
                    self.profile_count += 1

                self.last_received_time = time.time()  # mark data arrival
                self.has_collected_yet = True

                if self.profile_count % 100 == 0:
                    print(f"Cycle {self.current_cycle}: Saved {self.profile_count} profiles...")

    def _setup_communication(self):
        """Setup communication with the Keyence device."""
        # Open communication
        res = LJXAwrap.LJX8IF_EthernetOpen(self.device_id, self.ethernetConfig)
        print("EthernetOpen:", hex(res))
        if res != 0:
            raise RuntimeError("Failed to connect controller")

        # Init high-speed communication
        res = LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray(
            self.device_id,
            self.ethernetConfig,
            self.highspeed_port,
            self.my_callback,
            10,   # 0 = unlimited until stop
            0)
        print("InitHighSpeedDataCommunicationSimpleArray:", hex(res))
        if res != 0:
            raise RuntimeError("Init HS communication failed")

        # Pre-start
        req = LJXAwrap.LJX8IF_HIGH_SPEED_PRE_START_REQ()
        req.bySendPosition = 2
        res = LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication(
            self.device_id, req, self.profinfo)
        print("PreStartHighSpeedDataCommunication:", hex(res))
        if res != 0:
            raise RuntimeError("PreStart failed")

        self.xsize = self.profinfo.wProfileDataCount
        print(f"Xsize = {self.xsize} points per profile")

    def _start_collection(self):
        """Start the data collection process."""
        # Start high speed comm
        res = LJXAwrap.LJX8IF_StartHighSpeedDataCommunication(self.device_id)
        print("StartHighSpeedDataCommunication:", hex(res))
        if res != 0:
            raise RuntimeError("Start HS comm failed")

        # Start measure
        if not self.use_external_batchStart:
            LJXAwrap.LJX8IF_StartMeasure(self.device_id)

        print(f"Cycle {self.current_cycle}: Acquisition started. Auto-stops if no data for {self.data_timeout} seconds...")

    def _cleanup_communication(self):
        """Clean up communication resources."""
        try:
            LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(self.device_id)
            LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(self.device_id)
            LJXAwrap.LJX8IF_CommunicationClose(self.device_id)
        except:
            pass  # Ignore cleanup errors

    def _cleanup_csv(self):
        """Clean up CSV file resources."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def _reset_for_next_cycle(self):
        """Reset state for the next collection cycle."""
        self.profile_count = 0
        self.has_collected_yet = False
        self.last_received_time = time.time()

    def _run_single_cycle(self):
        """Run a single data collection cycle."""
        # Generate cycle-specific filename
        if self.auto_restart:
            base_name = self.csv_filename.replace('.csv', '')
            cycle_filename = f"{base_name}_cycle_{self.current_cycle:03d}.csv"
        else:
            cycle_filename = self.csv_filename

        # Open CSV file for this cycle
        self.csv_file = open(cycle_filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        try:
            # Setup communication
            self._setup_communication()
            
            # Start collection
            self._start_collection()

            # Wait for data collection to complete
            while self.running:
                time.sleep(0.1)
                if self.has_collected_yet:
                    if time.time() - self.last_received_time > self.data_timeout:
                        print(f"Cycle {self.current_cycle}: No data received for {self.data_timeout} seconds → stopping.")
                        break

        finally:
            # Cleanup
            self._cleanup_communication()
            self._cleanup_csv()
            
            print(f"Cycle {self.current_cycle}: Saved {self.profile_count} profiles to {cycle_filename}")
            self.total_profiles_collected += self.profile_count

    def run(self):
        """Run data collection with optional auto-restart."""
        print(f"Starting Keyence data collection...")
        if self.auto_restart:
            print(f"Auto-restart enabled with {self.restart_delay}s delay between cycles")
            if self.max_cycles:
                print(f"Maximum cycles: {self.max_cycles}")
            else:
                print("Unlimited cycles (use Ctrl+C to stop)")
        else:
            print("Single cycle mode")

        try:
            while self.running:
                self.current_cycle += 1
                
                # Check if we've reached max cycles
                if self.max_cycles and self.current_cycle > self.max_cycles:
                    print(f"Reached maximum cycles ({self.max_cycles}). Stopping.")
                    break

                print(f"\n=== Starting Collection Cycle {self.current_cycle} ===")
                
                # Run single cycle
                self._run_single_cycle()
                
                # Reset for next cycle
                self._reset_for_next_cycle()
                
                # Check if we should continue
                if self.auto_restart and self.running:
                    print(f"Waiting {self.restart_delay} seconds before next cycle...")
                    time.sleep(self.restart_delay)
                else:
                    break

        except KeyboardInterrupt:
            print("\nReceived Ctrl+C. Stopping gracefully...")
            self.running = False
        except Exception as e:
            print(f"Error during collection: {e}")
            self.running = False
        finally:
            # Final cleanup
            self._cleanup_communication()
            self._cleanup_csv()
            
            print(f"\n=== Collection Complete ===")
            print(f"Total cycles completed: {self.current_cycle}")
            print(f"Total profiles collected: {self.total_profiles_collected}")

    def stop(self):
        """Stop the collection process gracefully."""
        self.running = False


if __name__ == "__main__":
    acq = LJXAutoStopAcquisition(
        data_timeout=2.0,   # stop if no data for 2 sec
        csv_filename="scan_stream.csv",
        auto_restart=True,  # enable auto-restart
        restart_delay=2.0,  # 2 second delay between cycles
        max_cycles=None     # unlimited cycles (use Ctrl+C to stop)
    )
    acq.run()