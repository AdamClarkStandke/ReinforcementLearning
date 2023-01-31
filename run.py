"""
Simple orientation calculation from Accelerometer
"""
import pyb
import utime
from machine import I2C
from mpu6500 import MPU6500 
from libhw.sensor_buffer import SensorsBuffer
from libhw.postproc import PostPitchRoll
from libhw import servo
import micropython



SDA = "Y10"
SCL = "Y9"
PINS = ["B6", "B7", "A2","A3"]
INV = [True, False, True, False]
STACK_OBS = 4


def do_import(module_name):
    res = __import__("libhw.%s" % module_name,
                     globals(), locals(), [module_name])
    return res


def run(model_name):
    model = do_import(model_name)

    i2c = I2C(scl=SCL, sda=SDA)
    acc = MPU6500(i2c)
    buf = SensorsBuffer([acc], timer_index=1, freq=100,
                        batch_size=10, buffer_size=100)
    micropython.alloc_emergency_exception_buf(100)
    post = PostPitchRoll(buf, pad_yaw=True)
    buf.start()
    ch = servo.pins_to_timer_channels(PINS)
    brain = servo.ServoBrain()
    brain.init(ch, inversions=INV)

    obs = []
    obs_len = STACK_OBS*(3+4)
    frames = 0
    frame_time = 0
    ts = utime.ticks_ms()

    try:
        while True:
            for v in post:
                for n in brain.positions:
                    obs.append([n])
                for n in v:
                    obs.append([n])
                obs = obs[-obs_len:]
                if len(obs) == obs_len:
                    frames += 1
                    frame_time += utime.ticks_diff(utime.ticks_ms(), ts)
                    ts = utime.ticks_ms()
                    res = model.forward(obs)
                    pos = [v[0] for v in res]
                    print("%s, FPS: %.3f" % (pos, frames*1000/frame_time))
                    brain.positions = pos
    finally:
        buf.stop()
        brain.deinit()


