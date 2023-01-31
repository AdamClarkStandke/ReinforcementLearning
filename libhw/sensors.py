"""
Interface for sensors
"""


class SensorInitError(RuntimeError):
    pass


class Sensor:
    """
    Base sensor class
    """
    def __init__(self, *args, **kwargs):
        self._buffer = bytearray(len(self))
        mv = memoryview(self._buffer)
        self._byte = [mv[idx:idx+1] for idx in range(len(self))]

    @property
    def buffer(self):
        return self._buffer

    def __len__(self):
        """
        Returns length of raw buffer
        :return: integer
        """
        raise NotImplementedError

    def query(self, decoder):
        """
        Query the sensor, returns list of float values after decodings
        :param decoder: decoder function
        :return: list of floats
        """
        self.refresh()
        return decoder(self.buffer)

    def refresh(self):
        """
        Has to be overriden by concrete implementations to fill the buffer with data values
        """
        raise NotImplementedError
    
    @classmethod
    def decode(cls, b):
        """
        Decode bytes into float values
        Warning: cannot be called from interrupt handler, as floats allocate memory
        :return: list of floats
        """
        raise NotImplementedError
    
    
    def decode_s16(buf):
    	"""
    	Decode signed 16-bit value from bytebuffer
    	:param buf: bytebuffer
    	:return: int
    	"""
    	v = (buf[1] << 8) + buf[0]
    	if 0x8000 & v:
    		v = - (0x010000 - v)
    	return v
    	
    
    @classmethod
    def preprocess(cls, vals):
        """
        Preprocess list of float values into representation suitable for NNs
        :param vals: list of floats
        :return: list of floats
        """
        raise NotImplementedError
