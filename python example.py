# # Xeryon: Python Library
# A quick overview of the Xeryon python library.

#  !!! MAKE SURE THE CORRECT SETTINGS ARE SAVED ON THE DRIVE USING THE WINDOWS INTERFACE

# ## 1. Setup
# Note: the Xeryon.py file needs to be within the same folder.<br>
# pyserial library is required. "pip install pyserial" (! This is different from "pip install serial")
# The code below initializes the system.

from Xeryon import *
controller  = Xeryon("COM5", 115200)           # Setup serial communication
axisX       = controller.addAxis(Stage.XLS_312, "X") # Add all axis and specify the correct stage.

# You can find the axis name in the config.txt file that's provided with the stages.

# Now the system can be started. <br>
# In closed loop: first run axis_.findIndex() so the stage can find its index.

controller.start()

axisX.findIndex()
# For multiaxis systems, it's possible to do this in a loop:
# for axis in controller.getAllAxis():
#    axis.findIndex()

# # 2. Basic Control
# This includes the functions: setDPOS(), getDPOS(), getEPOS(), step(), setUnits() <br>
# DPOS stands for 'Desired POSition', this is the position that the stage trys to reach. <br>
# EPOS stands for 'Encoder POSition', this value indicates the actual position of the stage.<br>

axisX.setUnits(Units.mm)
axisX.setDPOS(5)
axisX.setDPOS(-5)

axisX.setUnits(Units.mu)
axisX.setDPOS(50)

print(axisX.getDPOS())
print(axisX.getEPOS())

axisX.setUnits(Units.mm)
axisX.setDPOS(0)
for i in range(0,5):
    axisX.step(1)    #Step 5x 1mm


# ## 3. Speed control and scanning
# Scanning: continiously move with fixed speed

axisX.setDPOS(0)
axisX.setUnits(Units.mm)
axisX.setSpeed(1)      # The speed value is set in the current unit. So here it is 1 mm/s.
axisX.startScan(-1)    # startScan() takes a negative or positive value.

axisX.stopScan()
axisX.setDPOS(0)      # Notice that the speed is still 1 mm/s.

axisX.startScan(1, 2)  # It is also possible to scan for a certain amount of seconds.


# ## 4. Getting data back from the controller
# The status bits: (defined in the datasheet)

axisX.isForceZero()
axisX.isMotorOn()
axisX.isClosedLoop()
axisX.isEncoderAtIndex()
axisX.isEncoderValid()
axisX.isSearchingIndex()
axisX.isPositionReached()
axisX.isEncoderError()
axisX.isScanning()
axisX.isAtLeftEnd()
axisX.isAtRightEnd()
axisX.isErrorLimit()
axisX.isSearchingOptimalFrequency()

# It is also possible to log all incoming data.<br>
# The function endLogging() returns a dictionary of the format: <br>
#     { "EPOS": [..,..,..], <br>
#       "DPOS": [..,..,..],<br>
#       "STAT": [..,..,..],<br>
#     ...}<br>

axisX.setUnits(Units.mm)
axisX.setSpeed(50)
axisX.setDPOS(0)
axisX.setSpeed(120)
axisX.startLogging()
axisX.step(0.01)
logs = axisX.endLogging()
axisX.setSpeed(50)
axisX.setDPOS(0)

from matplotlib import pyplot as plt

unit_converted_epos = []
for index in range(0, len(logs["EPOS"])):
    unit_converted_epos.append(axisX.convertEncoderUnitsToUnits(logs["EPOS"][index], axisX.units))

y = unit_converted_epos
plt.plot(y)
plt.ylabel('EPOS ('+str(axisX.units)+')')
plt.xlabel("Sample")
plt.title("EPOS")
plt.show()

controller.stop()