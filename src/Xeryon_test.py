#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:57:58 2024

@author: james
"""

import threading
from Xeryon import *
import serial.tools.list_ports
from time import sleep



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
    stage1.setSpeed(5)
    stage1.setUnits(Units.mu)
    stage1.setDPOS(-15000)
    sleep(2)
    

def func02():
    stage2.findIndex(forceWaiting = True)
    sleep(5)
    stage2.setSpeed(5)
    stage2.setUnits(Units.mu)
    stage2.setDPOS(-15000)
    sleep(2)

# TODO: make sure that stages are NOT attached to the membrane while find_Index() is running



def func1(position):
#     print("func1: starting")
    # stage1.startScan(1)
    # sleep(0.05)
    # stage1.stopScan()
    
    stage1.setDPOS(position)
#     print("func1: ending")
    
def func2(position):
#     print("func2: starting")
#     stage2.startScan(1)
#     sleep(0.05)
#     stage2.stopScan()
    
    stage2.setDPOS(position)
#     print("func2: ending")    


thread01 = threading.Thread(target = func01)
thread02 = threading.Thread(target = func02)

thread1 = threading.Thread(target = func1)
thread2 = threading.Thread(target = func2)

input("\nPress Enter to start homing...\n")

thread01.start()
thread02.start()
thread01.join()
thread02.join()

print("\nHoming complete. You may now mount the aperture.", end = '\n')

try:
    while True:
        position_input = input("\nInput position command [mm]: ")
        position = float(position_input)*1000
        
        
        if not -17000 < position < 17000:
            print("\nError: not a valid nozzle position", end = '\n')
            continue
            
        sleep(1)

        thread1 = threading.Thread(target = func1, args = (position,))
        thread2 = threading.Thread(target = func2, args = (position,))

        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()

except KeyboardInterrupt:
    print("\nUser quit program.", end = '\n \n')
    
finally:
    controller1.stop()
    controller2.stop()
    print("\nEnding program.", end = '\n \n')


