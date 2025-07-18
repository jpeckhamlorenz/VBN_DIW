#!/usr/bin/env python3

# from Xeryon import *
# controller  = Xeryon("COM5", 115200)           # Setup serial communication
# axisX       = controller.addAxis(Stage.XLS_312, "X") # Add all axis and specify the correct stage.

from time import sleep
import pigpio
import rospy
from std_msgs.msg import Float32, Float32MultiArray
import os
print("Ayo")
print(os.environ.get("ROS_MASTER_URI"))
print(os.environ.get("ROS_IP"))
steps_per_uL = 409
start_speed = 50 #mm/s
import threading
from Xeryon import *
import serial.tools.list_ports 


# Xeryon SETUP CODE
##############################################
##############################################
port1 = next(serial.tools.list_ports.grep('ACM0'))
controller1  = Xeryon(port1.device, 115200)           # Setup serial communication

    # controller = Xeryon("usbmodem11301", 115200)
    # arduino = serial.Serial(port.device,115200,timeout=1)

port2 = next(serial.tools.list_ports.grep('ACM1'))
controller2  = Xeryon(port2.device, 115200)


stage1 = controller1.addAxis(Stage.XLA_1250, "1") # Add all axis and specify the correct stage.
stage2 = controller2.addAxis(Stage.XLA_1250, "2")

controller1.start()
controller2.start()

def func01():
    stage1.findIndex(forceWaiting = True)
    sleep(5)
    stage1.setSpeed(start_speed)
    stage1.setUnits(Units.mu)
    stage1.setDPOS(-15000)
    sleep(2)

def func02():
    stage2.findIndex(forceWaiting = True)
    sleep(5)
    stage2.setSpeed(start_speed)
    stage2.setUnits(Units.mu)
    stage2.setDPOS(-15000)
    sleep(2)
    
def func1(position, speed):
    stage1.setSpeed(speed)  # mu/s
    stage1.setDPOS(position)
    
def func2(position, speed):
    stage2.setSpeed(speed)  # mu/s
    stage2.setDPOS(position)
    
thread01 = threading.Thread(target = func01)
thread02 = threading.Thread(target = func02)

thread1 = threading.Thread(target = func1)
thread2 = threading.Thread(target = func2)

input("\nPress Enter to start homing...\n")

thread01.start()
thread02.start()
thread01.join()
thread02.join()
sleep(4)

print("\nHoming complete. You may now mount the aperture.", end = '\n')
##############################################
##############################################

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
    
    assert 0.0 <= abs(msg.data) <= 15.0, "Flowrate Command is out of bounds"
    Q = msg.data
    
    log_message = "Flowrate Command: %s" %Q
    rospy.loginfo(log_message)  # writes output to terminal

    pi.set_PWM_dutycycle(stepA, 128)
    pi.set_PWM_frequency(stepA, int(abs(steps_per_uL*Q)))
    
    pi.set_PWM_dutycycle(stepB, 128)
    pi.set_PWM_frequency(stepB, int(abs(steps_per_uL*Q)))
    
    if Q > 0:
        pi.write(dirA, 0)
        pi.write(dirB, 0)
    else: 
        pi.write(dirA, 1)
        pi.write(dirB, 1)
       
       
def beadwidth_command_callback(msg):

    W_data = msg.data
    W_bead = float(W_data[0])
    W_speed = float(W_data[1])
    
    rospy.loginfo(W_data)

    bead_message = "Beadwidth Command: %s" % W_bead
    rospy.loginfo(bead_message)  # writes output to terminal

    speed_message = "Beadspeed Command: %s" % W_speed
    rospy.loginfo(speed_message)  # writes output to terminal

    calibration = [6596, -10954]
    W = calibration[0] * W_bead + calibration[1]
    
    if W_speed*calibration[0] > 100*1000:
        W_vel = 100*calibration[0]
    else:
        W_vel = W_speed * calibration[0]
        
    
    if not -17000 < W < 17000:
        rospy.logerror("\nError: not a valid nozzle position", end = '\n')
    else:
        log_message = "Nozzle Width Command: %s" %W
        
        rospy.loginfo(log_message)  # writes output to terminal
        thread1 = threading.Thread(target = func1, args = (W,W_vel))
        thread2 = threading.Thread(target = func2, args = (W,W_vel))


        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
    
def listener():

    rospy.init_node('VBN')
    rospy.Subscriber('flowrate_command', Float32, flowrate_command_callback, queue_size=1)
    rospy.Subscriber('beadwidth_command', Float32MultiArray, beadwidth_command_callback, queue_size=1)
    print("We are gooning. (Subscribed to flowrate/beadwidth/beadspeed)")
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



