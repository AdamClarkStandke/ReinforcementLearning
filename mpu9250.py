# Copyright (c) 2018-2020 Mika Tuupola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of  this software and associated documentation files (the "Software"), to
# deal in  the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copied of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://github.com/tuupola/micropython-mpu9250

"""
MicroPython I2C driver for MPU9250 9-axis motion tracking device
"""

# pylint: disable=import-error
from micropython import const
from mpu6500 import MPU6500, SF_G, SF_DEG_S
from ak8963 import AK8963
from libhw import sensors
import math as m
# pylint: enable=import-error

__version__ = "0.3.0"

# Used for enabling and disabling the I2C bypass access
_INT_PIN_CFG = const(0x37)
_I2C_BYPASS_MASK = const(0b00000010)
_I2C_BYPASS_EN = const(0b00000010)
_I2C_BYPASS_DIS = const(0b00000000)

class MPU9250:
    """Class which provides interface to MPU9250 9-axis motion tracking device."""
    def __init__(self, i2c, mpu6500 = None, ak8963 = None, addr=0x68):
    	
        if mpu6500 is None:
            self.mpu6500 = MPU6500(i2c, accel_sf=SF_G, gyro_sf=SF_DEG_S)
        else:
            self.mpu6500 = mpu6500

        # Enable I2C bypass to access AK8963 directly.
        char = self.mpu6500._register_char(_INT_PIN_CFG)
        char &= ~_I2C_BYPASS_MASK # clear I2C bits
        char |= _I2C_BYPASS_EN
        self.mpu6500._register_char(_INT_PIN_CFG, char)


    @property
    def acceleration(self):
        """
        Acceleration measured by the sensor. By default will return a
        3-tuple of X, Y, Z axis values in m/s^2 as floats. To get values in g
        pass `accel_fs=SF_G` parameter to the MPU6500 constructor.
        """
        return self.mpu6500.acceleration
        
    
    def __len__(self):
        return 6
     
       
    @classmethod
    def decode(cls, b):
       so = self.mpu6500.so
       sf = self.mpu6500.sf
       return [self.decode_s16(b[ofs:ofs+2])/so*sf for ofs in range(0, 6, 2)]

   
    @classmethod
    def preprocess(cls, vals):
        """
        Preprocess list of float values into representation suitable for NNs
        :param vals: list of floats
        :return: list of floa
        """
        l = m.sqrt(vals[0]*vals[0]+vals[1]*vals[1]+vals[2]*vals[2])
        if abs(l) > 1e-5:
           c = (l-1)/l
           return [c*v for v in vals] 
        return vals
        
    def twosComp(x):
    	if 0x8000 & x:
    		x = -(0x010000-x)
    	return x
    	
    def decode_s16(buf):
    	v=(buf[1] << 8) + buf[0] 
    	return twosComp(v)
    	
    @property
    def whoami(self):
        return self.mpu6500.whoami

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
