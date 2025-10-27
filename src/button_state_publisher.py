#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
import pigpio

BUTTON_GPIO = 20
 
if __name__ == '__main__':
    rospy.init_node('button_state_publisher')
    
    pub = rospy.Publisher('button_state', Bool, queue_size = 10)
    
    pi = pigpio.pi()
    
    pi.set_mode(BUTTON_GPIO, pigpio.INPUT)
    pi.set_pull_up_down(BUTTON_GPIO,pigpio.PUD_UP)

    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        gpio_state = not pi.read(BUTTON_GPIO)
#         rospy.loginfo(gpio_state)  # writes output to terminal
        pub.publish(gpio_state)
        rate.sleep()
    
