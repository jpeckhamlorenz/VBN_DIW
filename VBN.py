#!/usr/bin/env python3

from Xeryon import *


from time import sleep
import pigpio
import rospy
from std_msgs.msg import Float32

DISABLE_WAITING = True

controller  = Xeryon("COM5", 115200)           # Setup serial communication
left_stage       = controller.addAxis(Stage.XLA_78, "Z") # Add all axis and specify the correct stage.
right_stage = controller.addAxis(Stage.XLA_78, "Z")

controller.start()


# TODO: make sure that stages are NOT attached to the membrane while find_Index() is running
for stage in controller.getAllAxis():
    stage.findIndex(forceWaiting = True)
    sleep(5)

for stage in controller.getAllAxis():
    stage.setUnits(Units.mu)
    stage.setSpeed(10)


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


def membrane_dilation_converter(requested_diam):
    
    actuator_DPOS_command = (
        0*requested_diam**0+
        1*requested_diam**1 + 
        0*requested_diam**2+
        0*requested_diam**3+
        0*requested_diam**4+
        0*requested_diam**5)
    
    return actuator_DPOS_command

def flowrate_command_callback(msg):
    
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
    
    stage_DPOS_command = membrane_dilation_converter(W)
    
    for stage in controller.getAllAxis(): 
        stage.setDPOS(stage_DPOS_command)  # [um]
    
    
def listener():
    
    rospy.init_node('VBN')
    rospy.Subscriber('flowrate_command', Float32, flowrate_command_callback)
    rospy.Subscriber('beadwidth_command', Float32, beadwidth_command_callback)
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
