from libhw.sensor_buffer import SensorsBuffer
from machine import I2C
from mpu6500 import MPU6500
import micropython


i2c = I2C(scl="Y9", sda="Y10")
print(i2c.scan())
print(hex(ord(i2c.readfrom_mem(104, 0x0F, 1))))
accel = MPU6500(i2c)
buf = SensorsBuffer([accel], 1, freq=100, batch_size=10, buffer_size=10)
micropython.alloc_emergency_exception_buf(100)
buf.start()
for b in buf:
	for v in b:
		data = accel.decode(v)
		print(data)
