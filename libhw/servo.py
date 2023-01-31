import pyb


class ServoBrain:
    FREQ = 50               # 20ms -- standard pulse interval for servos
    MIN_PERCENT = 2.3
    MAX_PERCENT = 12.7
    MIN_POS = 0
    MAX_POS = 1

    def __init__(self):
        """
        Construct servo brain
        """
        self._timer_channels = None
        self._inversions = None
        self._channels = None
        self._positions = None
        self._timers = None

    def init(self, timer_channels, inversions=None):
        """
        :param timer_channels: list of tuples (pin_name, (timer, channel))
        :param inversions: list of bools specifying servos to be inverted, if None, no inversions
        """
        self._timer_channels = timer_channels
        self._inversions = inversions if inversions is not None else [False for _ in timer_channels]
        self._timers = {}
        self._channels = []
        self._positions = []
        for pin_name, (t_idx, ch_idx) in timer_channels:
            if t_idx not in self._timers:
                self._timers[t_idx] = pyb.Timer(t_idx, freq=self.FREQ)
            pin = pyb.Pin(pin_name, pyb.Pin.OUT)
            self._channels.append(self._timers[t_idx].channel(ch_idx, pyb.Timer.PWM, pin=pin))
            self._positions.append(0.0)
        self._apply_positions(self._positions)

    def deinit(self):
        if self._timers is None:
            return
        for t in self._timers.values():
            t.deinit()
        self._timer_channels = None
        self._channels = None
        self._positions = None
        self._timers = None

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value
        self._apply_positions(self._positions)

    @classmethod
    def position_to_percent(cls, pos):
        return (pos-cls.MIN_POS)*(cls.MAX_PERCENT - cls.MIN_PERCENT)/(cls.MAX_POS - cls.MIN_POS) + cls.MIN_PERCENT

    def _apply_positions(self, values):
        for p, ch, inv in zip(values, self._channels, self._inversions):
            if inv:
                p = self.MAX_POS - p
            ch.pulse_width_percent(self.position_to_percent(p))


_PINS_TO_TIMER = {
    "B6": (4, 1),
    "B7": (4, 2),
    "B8": (4, 3),
    "B9": (4, 4),
    "A2": (2, 3),
    "A3": (2, 4),
}


def pins_to_timer_channels(pins):
    """
    Convert list of pins to list of tuples (timer, channel). This function is hardware-specific
    :param pins: list of pin names
    :return: list of (pin_name, (timer, channel)) tuples
    """
    res = []
    for p in pins:
        pair = _PINS_TO_TIMER.get(p)
        assert pair is not None
        res.append((p, pair))
    return res


