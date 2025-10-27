#!/usr/bin/env python3

import time
import pigpio
import rospy
from std_msgs.msg import Float32, Float32MultiArray
import os
print(os.environ.get("ROS_MASTER_URI"))
print(os.environ.get("ROS_IP"))
import threading


# %%

class KeyenceScan:
    def __init__(self, GPIO_PIN: int = 18, scan_freq: int = 100,
                 control_freq: float = 20.0, stale_timeout: float = 2.0, debounce_buffer: float = 0.05,
                 auto_reset_delay: float = 1.0):
        self.GPIO_PIN = GPIO_PIN
        self.SCAN_FREQ = scan_freq
        self.STALE_TIMEOUT = stale_timeout
        self.DEBOUNCE_BUFFER = debounce_buffer
        self.AUTO_RESET_DELAY = auto_reset_delay

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running (try: sudo systemctl start pigpiod)")
        self.pi.set_mode(self.GPIO_PIN, pigpio.OUTPUT)

        self.desired = False
        self.current = False
        self.lock = threading.Lock()
        self.last_change = 0.0
        self.last_msg = time.time()
        self.last_stop_time = 0.0
        self.auto_reset_enabled = True

        self.timer = rospy.Timer(rospy.Duration(1/control_freq), self._tick)  # 20 Hz control loop
        rospy.on_shutdown(self.shutdown)

    def _start_scanning(self, duty: float = 0.5):

        # typically 50% (expressed in millionths) duty cycle square wave at desired frequency
        self.pi.hardware_PWM(self.GPIO_PIN, self.SCAN_FREQ, int(duty * 1e6))
        self.current = True
        rospy.loginfo("Scan PWM: START (freq=%d Hz, duty=%.1f%%)", self.SCAN_FREQ, duty * 100)

    def _stop_scanning(self):
        self.pi.hardware_PWM(self.GPIO_PIN, 0, 0)
        self.current = False
        self.last_stop_time = time.time()
        rospy.loginfo("Scan PWM: STOP")

    def _reset_for_next_cycle(self):
        """Reset the scanner state to be ready for the next scan cycle"""
        with self.lock:
            self.desired = False
            self.current = False
            self.last_change = 0.0
            self.last_stop_time = 0.0
        rospy.loginfo("Scanner reset and ready for next cycle")

    def _scan_callback(self, msg: Float32):
        print("Scan Cummand:", msg.data)
        with self.lock:
            self.desired = bool(msg.data)
            self.last_msg = time.time()

    def _tick(self, event):
        now = time.time()
        with self.lock:
            # Auto-reset logic: if scanning was stopped and enough time has passed, reset for next cycle
            if (self.auto_reset_enabled and 
                self.last_stop_time > 0 and 
                not self.current and 
                not self.desired and 
                (now - self.last_stop_time) > self.AUTO_RESET_DELAY):
                # Reset for next cycle
                self._reset_for_next_cycle()
                return

            # Watchdog: if stale, force OFF
            if (now - self.last_msg) > self.STALE_TIMEOUT:
                self.desired = False

            # Debounce: avoid rapid toggling
            if (now - self.last_change) < self.DEBOUNCE_BUFFER:
                return

            # Transition logic
            if self.desired and not self.current:
                # Start PWM only once on rising command
                self._start_scanning()
                self.last_change = now

            elif (not self.desired) and self.current:
                # Stop PWM only once on falling command
                self._stop_scanning()
                self.last_change = now

    def set_auto_reset(self, enabled: bool):
        """Enable or disable automatic reset after scanning stops"""
        self.auto_reset_enabled = enabled
        rospy.loginfo("Auto-reset %s", "enabled" if enabled else "disabled")

    def listener(self):
        rospy.Subscriber('/scan_command', Float32, self._scan_callback)
        rospy.loginfo("scanner node up. Waiting for /scan_command (Float32) msgs...")
        rospy.loginfo("Auto-reset enabled with %.1f second delay", self.AUTO_RESET_DELAY)
        rospy.spin()

    def shutdown(self):
        # Ensure PWM is off on exit
        self.pi.hardware_PWM(self.GPIO_PIN, 0, 0)
        self.pi.stop()


# %%

if __name__ == "__main__":
    rospy.init_node('scanner')
    ScanNode = KeyenceScan()
    ScanNode.listener()
