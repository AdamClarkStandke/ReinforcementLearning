"""
Simple orientation calculation from Accelerometer
"""
import pyb
from machine import I2C
from mpu6500 import MPU6500
from libhw.sensor_buffer import SensorsBuffer
from libhw.postproc import PostPitchRoll
import micropython

SDA = "Y10"
SCL = "Y9"


def run():
    i2c = I2C(scl="Y9", sda="Y10")
    # get acceleration data
    accel = MPU6500(i2c)
    buf = SensorsBuffer([accel], 1, freq=100, batch_size=10, buffer_size=10)
    micropython.alloc_emergency_exception_buf(100)
    post = PostPitchRoll(buf, pad_yaw=True)
    buf.start()
    try:
        while True:
            for v in post:
                print("pitch=%s, roll=%s, yaw=%s" % tuple(v))
    finally:
        buf.stop()

