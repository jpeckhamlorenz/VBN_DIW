#!/usr/bin/env python3

import rospy
import pigpio
from std_msgs.msg import Bool

LED_GPIO = 21

def button_state_callback(msg):
	pi.write(LED_GPIO, msg.data)

if __name__ == '__main__':
	rospy.init_node('led_actuator')
	
	pi = pigpio.pi()
	
	pi.set_mode(LED_GPIO, pigpio.OUTPUT)
	
	rospy.Subscriber('button_state', Bool, button_state_callback)
	
	rospy.spin()
