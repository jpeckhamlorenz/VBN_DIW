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
                 control_freq: float = 20.0, stale_timeout: float = 2.0, debounce_buffer: float = 0.05):
        self.GPIO_PIN = GPIO_PIN
        self.SCAN_FREQ = scan_freq
        self.STALE_TIMEOUT = stale_timeout
        self.DEBOUNCE_BUFFER = debounce_buffer

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running (try: sudo systemctl start pigpiod)")
        self.pi.set_mode(self.GPIO_PIN, pigpio.OUTPUT)

        self.desired = False
        self.current = False
        self.lock = threading.Lock()
        self.last_change = 0.0
        self.last_msg = time.time()

        self.timer = rospy.Timer(rospy.Duration(1/control_freq), self._tick)  # 20 Hz control loop
        rospy.on_shutdown(self.shutdown)

    def start_scanning(self, duty: float = 0.5):

        # typically 50% (expressed in millionths) duty cycle square wave at desired frequency
        self.pi.hardware_PWM(self.GPIO_PIN, self.SCAN_FREQ, int(duty * 1e6))
        self.current = True
        rospy.loginfo("Scan PWM: START (freq=%d Hz, duty=%.1f%%)", self.SCAN_FREQ, duty * 100)

    def _stop_scanning(self):
        self.pi.hardware_PWM(self.GPIO_PIN, 0, 0)
        self.current = False
        rospy.loginfo("Scan PWM: STOP")

    def _scan_callback(self, msg: Float32):
        with self.lock:
            self.desired = bool(msg.data)
            self.last_msg = time.time()

    def _tick(self, event):
        now = time.time()
        with self.lock:
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

    def listener(self):
        rospy.init_node('scanner')
        rospy.Subscriber('scan_command', Float32, self.scan_callback)
        rospy.loginfo("scanner node up. Waiting for /scan_command (Float32) msgs...")
        rospy.spin()

    def shutdown(self):
        # Ensure PWM is off on exit
        self.pi.hardware_PWM(self.GPIO_PIN, 0, 0)
        self.pi.stop()


# %%

if __name__ == "__main__":
    ScanNode = KeyenceScan()
    ScanNode.listener()
