#!/usr/bin/env python3

# from Xeryon import *
# controller  = Xeryon("COM5", 115200)           # Setup serial communication
# axisX       = controller.addAxis(Stage.XLS_312, "X") # Add all axis and specify the correct stage.

from time import sleep
import pigpio
import rospy
from std_msgs.msg import Float32
import os
print("Ayo")
print(os.environ.get("ROS_MASTER_URI"))
print(os.environ.get("ROS_IP"))
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

def flowrate_command_callback(msg):
    assert 0.0 <= msg.data <= 3.0, "Flowrate Command is out of bounds"
    Q = msg.data
    
    log_message = "Flowrate Command: %s" %Q
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
       
       
def beadwidth_command_callback(msg):
    W = msg.data
    
    log_message = "Nozzle Width Command: %s" %W
    rospy.loginfo(log_message)  # writes output to terminal
    
    #TODO: something something make VBN actuator go brr (wrapper time?)
    
    
def listener():
    
    rospy.init_node('VBN')
    rospy.Subscriber('flowrate_command', Float32, flowrate_command_callback)
    # rospy.Subscriber('beadwidth_command', Float32, beadwidth_command_callback)
    rospy.spin()
    
    
if __name__ == '__main__':
    try:
        listener()
    except KeyboardInterrupt:
        print("user quit program")
    finally:
        pi.set_PWM_dutycycle(stepA, 0)
        pi.set_PWM_dutycycle(stepB, 0)
        pi.write(enA, 1)
        pi.write(enB, 1)


