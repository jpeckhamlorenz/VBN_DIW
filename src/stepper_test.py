from time import sleep
import pigpio

enA = 27
dirA = 17
stepA = 12

enB = 6
dirB = 16
stepB = 13



pi = pigpio.pi()

pi.set_mode(enA, pigpio.OUTPUT)
pi.set_mode(dirA, pigpio.OUTPUT)
pi.set_mode(stepA, pigpio.OUTPUT)

pi.set_mode(enB, pigpio.OUTPUT)
pi.set_mode(dirB, pigpio.OUTPUT)
pi.set_mode(stepB, pigpio.OUTPUT)

powah = 400

pi.set_PWM_dutycycle(stepA, 128)
pi.set_PWM_frequency(stepA, powah)

pi.set_PWM_dutycycle(stepB, 128)
pi.set_PWM_frequency(stepB, powah)

pi.write(enA,0)
pi.write(enB,0)

try:
    while True:
#         pi.write(dirA, 1)
#         pi.write(dirB, 1)
#         sleep(2)
        pi.write(dirA, 0)
        pi.write(dirB, 0)
        sleep(2)
except KeyboardInterrupt:
    print("user quit program")
finally:
    pi.set_PWM_dutycycle(stepA, 0)
    pi.set_PWM_dutycycle(stepB, 0)
    pi.write(enA, 1)
    pi.write(enB, 1)
