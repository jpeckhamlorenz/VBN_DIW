#!/usr/bin/env python3

from time import sleep
import pigpio
import rospy
from std_msgs.msg import Float32

steps_per_uL = 409

enA = 27
dirA = 17
stepA = 18

enB = 13
dirB = 16  
stepB = 19

pi = pigpio.pi()

pi.set_mode(enA, pigpio.OUTPUT)
pi.set_mode(dirA, pigpio.OUTPUT)
pi.set_mode(stepA, pigpio.OUTPUT)

pi.set_mode(enB, pigpio.OUTPUT)
pi.set_mode(dirB, pigpio.OUTPUT)
pi.set_mode(stepB, pigpio.OUTPUT)


pi.write(enA,0)
pi.write(enB,0)

def VBN_command_callback(msg):
    
    data = msg.data
    Q = data[0]
    W = data[1]
    
    log_message = "Flow Command: %s  and  Nozzle Command: %s" % (Q, W)
    rospy.loginfo(log_message)  # writes output to terminal

    pi.set_PWM_dutycycle(stepA, 128)
    pi.set_PWM_frequency(stepA, int(abs(steps_per_uL*Q)))
    
    pi.set_PWM_dutycycle(stepB, 128)
    pi.set_PWM_frequency(stepB, int(abs(steps_per_uL*Q)))
    
    if Q > 0:
        pi.write(dirA, 0)
        pi.write(dirB, 1)
    else: 
        pi.write(dirA, 1)
        pi.write(dirB, 0)

if __name__ == '__main__':
    try:
        rospy.init_node('VBN')
        rospy.Subscriber('VBN_command', MSG_TYPE, VBN_command_callback)  # TODO
        rospy.spin()
    except KeyboardInterrupt:
        print("user quit program")
    finally:
        pi.set_PWM_dutycycle(stepA, 0)
        pi.set_PWM_dutycycle(stepB, 0)
        pi.write(enA, 1)
        pi.write(enB, 1)
