"""Sensor hub.

READING THIS FILE
Each sensor is wrapped in its own small class. Every class follows the same pattern:
  - try to open the device,
  - on success, keep a handle,
  - on failure, print one line and set the handle to None.

That is why the rover still boots with a missing sensor: every caller checks for None
and carries on. The web UI shows a dash instead of a number.

Units: distances in metres, temperatures in °C, headings in degrees clockwise from
north, voltages in volts, CO2 in ppm, substrate moisture in percent.
"""
import math                          # radians/degrees for the compass
import threading                     # the GPS parser runs in its own thread
import time                          # monotonic clocks for staleness checks
from pathlib import Path             # the DS18B20 is a file under /sys/bus/w1

import numpy as np                   # the thermal frame is a numpy array
import serial                        # pyserial, for both the GPS and the LiDAR

W1_DEVICES = Path("/sys/bus/w1/devices")   # where the kernel exposes 1-Wire probes


def _log(tag, message):
    """One-line boot message, e.g. [gnss] listening on /dev/ttyAMA0."""
    print(f"[{tag}] {message}", flush=True)   # flush so journalctl shows it immediately


class Gnss:
    """u-blox NMEA over UART, parsed in a background thread."""

    def __init__(self, cfg):
        self.cfg = cfg                            # port, baud, quality thresholds
        self.lock = threading.Lock()              # guards the fields below
        self.lat = None                           # last latitude, or None
        self.lon = None                           # last longitude, or None
        self.quality = 0                          # NMEA fix quality: 0 none, 1 GPS, 4 RTK
        self.sats = 0                             # satellites in use
        self.hdop = 99.0                          # dilution of precision, lower is better
        self.updated = 0.0                        # when the last fix arrived
        self.port = None                          # serial handle, if it opened
        try:
            import pynmea2                        # NMEA sentence parser
            self.nmea = pynmea2                   # keep it for the background thread
            self.port = serial.Serial(cfg["port"], cfg["baud"], timeout=2)  # open UART
            threading.Thread(target=self._run, daemon=True).start()         # start reading
            _log("gnss", f"listening on {cfg['port']}")                     # boot log
        except Exception as exc:                  # UART missing, permission denied, etc.
            _log("gnss", f"unavailable: {exc}")   # report and carry on without GPS

    def _run(self):
        """Background loop: read NMEA lines forever."""
        while True:                               # the thread lives as long as the rover
            try:
                line = self.port.readline().decode("ascii", errors="ignore").strip()  # one sentence
                if not line.startswith("$"):      # NMEA sentences always start with $
                    continue                      # junk or a blank line, skip it
                msg = self.nmea.parse(line)       # split into fields
                if msg.sentence_type not in ("GGA", "GNS"):   # we only want fix information
                    continue                      # ignore RMC, VTG, GSV and friends
                if getattr(msg, "gps_qual", 0) in (0, None):  # 0 = no valid fix
                    continue                      # do not publish a bad position
                with self.lock:                   # publish atomically
                    self.lat = float(msg.latitude)        # decimal degrees
                    self.lon = float(msg.longitude)       # decimal degrees
                    self.quality = int(msg.gps_qual)      # fix quality
                    self.sats = int(getattr(msg, "num_sats", 0) or 0)   # satellite count
                    self.hdop = float(getattr(msg, "hdop", 99.0) or 99.0)  # precision
                    self.updated = time.monotonic()       # timestamp of this fix
            except Exception:                     # bad sentence, unplugged cable, etc.
                time.sleep(0.2)                   # do not spin the CPU on errors

    def position(self):
        """Current (lat, lon), or None if we have never had a fix."""
        with self.lock:                           # read atomically
            if self.lat is None:                  # no fix yet
                return None                       # caller must handle this
            return (self.lat, self.lon)           # a plain tuple

    @property
    def fix_ok(self):
        """True only if the fix is fresh AND good enough to drive on."""
        with self.lock:                           # read all fields together
            fresh = time.monotonic() - self.updated < 5.0      # fix younger than 5 s
            return (fresh
                    and self.lat is not None                       # we have a position
                    and self.quality >= self.cfg["min_quality"]    # GPS or better
                    and self.hdop <= self.cfg["max_hdop"]          # precision good enough
                    and self.sats >= self.cfg["min_sats"])         # enough satellites

    def average_position(self, seconds):
        """Average fixes for `seconds` — used once, at arming, to set home."""
        samples = []                              # every good fix we collect
        end = time.monotonic() + seconds          # stop time
        while time.monotonic() < end:             # keep collecting until then
            if self.fix_ok:                       # only use trustworthy fixes
                samples.append(self.position())   # remember it
            time.sleep(0.2)                       # five samples per second is plenty
        if not samples:                           # nothing arrived
            return None                           # caller must not set home
        return (sum(p[0] for p in samples) / len(samples),   # mean latitude
                sum(p[1] for p in samples) / len(samples))   # mean longitude

    def quality_text(self):
        """Short label for the phone UI."""
        if not self.fix_ok:                       # no usable fix
            return "NO FIX"                       # display this
        return {1: "GPS", 2: "DGPS", 4: "RTK-FIX", 5: "RTK-FLOAT"}.get(self.quality, "FIX")


class Imu:
    """BNO085 heading, tilt-compensated and referenced to magnetic north."""

    def __init__(self, cfg):
        self.sensor = None                                     # handle, if the IMU opened
        self.declination = float(cfg.get("declination_deg", 0.0))  # magnetic -> true north
        try:
            import board                                       # Blinka pin names
            import busio                                       # Blinka I2C bus
            from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR   # fused orientation report
            from adafruit_bno08x.i2c import BNO08X_I2C          # I2C driver

            i2c = busio.I2C(board.SCL, board.SDA)               # open /dev/i2c-1
            self.sensor = BNO08X_I2C(i2c, address=cfg.get("i2c_address", 0x4A))  # talk to it
            self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)  # ask for orientation
            _log("imu", "BNO085 ready")                         # boot log
        except Exception as exc:                                # not wired, wrong address, etc.
            _log("imu", f"unavailable: {exc}")                  # carry on without heading

    def heading(self):
        """Degrees clockwise from true north, or None."""
        if not self.sensor:                                     # no IMU
            return None                                         # caller must handle it
        try:
            yaw = math.degrees(self.sensor.euler[2])            # euler = (roll, pitch, yaw)
        except Exception:                                       # I2C glitch
            return None                                         # treat as "no heading"
        return (yaw + self.declination + 360.0) % 360.0         # magnetic -> true, wrap to 0..360


class Thermal:
    """MLX90640 32x24 IR array."""

    ROWS, COLS = 24, 32                                         # sensor resolution

    def __init__(self, cfg, refresh_hz=8):
        self.mlx = None                                         # handle, if it opened
        try:
            import board                                        # Blinka pin names
            import busio                                        # Blinka I2C bus
            import adafruit_mlx90640                            # sensor driver

            i2c = busio.I2C(board.SCL, board.SDA)               # open /dev/i2c-1
            self.mlx = adafruit_mlx90640.MLX90640(i2c)          # talk to the array
            # Map the config number to the library's enum. Unknown values fall back to 8 Hz.
            rates = {1: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ,
                     2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
                     4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
                     8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
                     16: adafruit_mlx90640.RefreshRate.REFRESH_16_HZ}
            self.mlx.refresh_rate = rates.get(int(refresh_hz), rates[8])  # apply the rate
            self.buffer = [0.0] * (self.ROWS * self.COLS)       # 768 floats, reused every frame
            _log("thermal", "MLX90640 ready")                   # boot log
        except Exception as exc:                                # not wired or clock stretching
            _log("thermal", f"unavailable: {exc}")              # detector will report no data

    def read(self):
        """Return a (24, 32) float array of °C, or None."""
        if not self.mlx:                                        # no sensor
            return None                                         # caller must handle it
        for _ in range(3):                                      # retry transient I2C errors
            try:
                self.mlx.getFrame(self.buffer)                   # fill the 768-float buffer
                return np.array(self.buffer, dtype=float).reshape(self.ROWS, self.COLS)  # 24x32
            except Exception:                                   # CRC error, bus glitch
                time.sleep(0.02)                                # brief pause, then retry
        return None                                             # three failures: no frame


class Lidar:
    """TF-Luna 9-byte frames, byte-synchronised on 0x59 0x59."""

    def __init__(self, cfg):
        self.ser = None                                     # serial handle, if it opened
        self._buf = b""                                     # leftover bytes between reads
        try:
            self.ser = serial.Serial(cfg["port"], cfg["baud"], timeout=0.05)  # open USB serial
            _log("lidar", f"listening on {cfg['port']}")    # boot log
        except Exception as exc:                             # adapter missing, wrong port, etc.
            _log("lidar", f"unavailable: {exc}")            # obstacle stop becomes unavailable

    def read_m(self):
        """Return the distance in metres, or None if no valid frame arrived in 250 ms."""
        if not self.ser:                                    # no LiDAR
            return None                                     # caller treats this as "clear"
        deadline = time.monotonic() + 0.25                  # do not block forever
        while time.monotonic() < deadline:                  # try until the deadline
            chunk = self.ser.read(1)                        # read one byte at a time
            if not chunk:                                   # nothing arrived
                continue                                    # keep waiting
            self._buf += chunk                              # append to the working buffer
            while len(self._buf) >= 9:                      # a TF-Luna frame is 9 bytes
                if self._buf[0] == 0x59 and self._buf[1] == 0x59:   # frame header
                    frame = self._buf[:9]                   # candidate frame
                    if (sum(frame[:8]) & 0xFF) == frame[8]: # checksum: low byte of the sum
                        self._buf = self._buf[9:]           # consume the frame
                        return (frame[2] | (frame[3] << 8)) / 100.0  # little-endian cm -> m
                self._buf = self._buf[1:]                   # wrong header: drop one byte and retry
        return None                                         # timed out


class Co2:
    """Sensirion SCD41 CO2 sensor on I2C 0x62.

    The sensor runs its own 5 s measurement cycle. Reading it faster does not make it
    update faster, so `read()` is gated to once a second and returns the last good
    value in between. Early firmware returns 0 before the first conversion, which is
    treated as "no reading" rather than 0 ppm."""

    def __init__(self, cfg):
        self.scd = None                                     # handle, if it opened
        self.last = None                                    # last good ppm value
        self._updated = 0.0                                 # when `last` was refreshed
        self._last_poll = 0.0                               # last attempt, to rate-limit I2C
        try:
            import board                                    # Blinka pin names
            import busio                                    # Blinka I2C bus
            import adafruit_scd4x                           # SCD40/SCD41 driver

            i2c = busio.I2C(board.SCL, board.SDA)           # open /dev/i2c-1
            self.scd = adafruit_scd4x.SCD4X(i2c, address=cfg.get("i2c_address", 0x62))
            self.scd.start_periodic_measurement()           # let it measure on its own clock
            _log("co2", "SCD41 ready")                      # boot log
        except Exception as exc:                            # not wired, wrong address, etc.
            _log("co2", f"unavailable: {exc}")              # CO2 gate is disabled

    def read(self):
        """Latest CO2 in ppm, or None if the sensor is missing or stale."""
        if not self.scd:                                    # no sensor
            return None                                     # caller skips the CO2 gate
        now = time.monotonic()                              # one clock for both checks
        if now - self._last_poll < 1.0:                     # asked too recently
            return self._fresh(now)                         # serve the cached value
        self._last_poll = now                               # this attempt counts
        try:
            ppm = int(self.scd.CO2)                         # cached measurement in ppm
            if ppm > 0:                                     # 0 means "not ready yet"
                self.last = ppm                             # keep it
                self._updated = now                         # it is fresh now
        except Exception:                                   # I2C glitch or data not ready
            pass                                            # keep the previous value
        return self._fresh(now)                             # None if stale beyond 30 s

    def _fresh(self, now):
        """Return the last value only if it is younger than 30 s."""
        if self.last is None or now - self._updated > 30.0:  # never read, or sensor died
            return None                                     # treat as unavailable
        return self.last                                    # a usable number


class Temperature:
    """DS18B20 1-Wire probe, read straight from the kernel's sysfs file.

    No Python library is needed: the kernel writes each conversion to
    /sys/bus/w1/devices/28-<serial>/w1_slave as two lines of text. The probe converts
    every 750 ms, so reads are cached for a second."""

    def __init__(self, cfg):
        self.path = None                                    # sysfs file, if one was found
        self.last = None                                    # last good °C
        self._updated = 0.0                                 # when `last` was refreshed
        self._last_poll = 0.0                               # rate-limit file reads
        try:
            wanted = (cfg or {}).get("sensor_id") or ""     # optional fixed serial
            if wanted:                                      # use the probe you named
                self.path = W1_DEVICES / wanted / "w1_slave"
            else:                                           # otherwise take the first one
                found = sorted(W1_DEVICES.glob("28-*"))     # one folder per probe
                if found:                                   # at least one is wired
                    self.path = found[0] / "w1_slave"       # use it
            if self.path is None:                           # nothing on the bus
                raise FileNotFoundError(
                    "no 28-* device (is dtoverlay=w1-gpio enabled?)")
            _log("temp", f"DS18B20 on {self.path.parent.name}")   # boot log
        except Exception as exc:                            # overlay missing, bad serial, etc.
            self.path = None                                # leave it disabled
            _log("temp", f"unavailable: {exc}")             # thermal ΔT check is disabled

    def read(self):
        """Reference temperature in °C, or None if missing or stale."""
        if self.path is None:                               # no probe
            return None                                     # caller skips the ΔT gate
        now = time.monotonic()                              # one clock for both checks
        if now - self._last_poll < 1.0:                     # read at most once a second
            return self._fresh(now)                         # cached value
        self._last_poll = now                               # this attempt counts
        try:
            lines = self.path.read_text(encoding="ascii").splitlines()  # two short lines
            if lines and lines[0].strip().endswith("YES"):  # CRC good = trustworthy value
                for part in lines[1].split():               # "… t=23500"
                    if part.startswith("t="):               # the temperature field
                        self.last = int(part[2:]) / 1000.0  # millidegrees -> °C
                        self._updated = now                 # mark it fresh
                        break                               # done with this read
        except Exception:                                   # unplugged or unreadable
            pass                                            # keep the previous value
        return self._fresh(now)                             # None if stale beyond 60 s

    def _fresh(self, now):
        """Return the last value only if it is younger than 60 s."""
        if self.last is None or now - self._updated > 60.0:  # never read, or probe died
            return None                                     # treat as unavailable
        return self.last                                    # a usable number


class Moisture:
    """SEN0193 capacitive moisture probe through an ADS1115 ADC on I2C 0x48.

    The probe is analog — its output falls from about `dry_v` in dry sand to `wet_v`
    when saturated — and a Raspberry Pi has no analog input, so the ADS1115 converts
    it. `read()` maps the voltage onto 0-100 % using those two calibration points."""

    def __init__(self, cfg):
        self.chan = None                                    # ADC channel, if it opened
        self.dry = float(cfg.get("dry_v", 2.45))            # volts in dry sand
        self.wet = float(cfg.get("wet_v", 1.30))            # volts in saturated sand
        self.last = None                                    # last good percentage
        self._last_poll = 0.0                               # rate-limit ADC reads
        try:
            import board                                    # Blinka pin names
            import busio                                    # Blinka I2C bus
            import adafruit_ads1x15.ads1115 as ADS          # 4-channel 16-bit ADC
            from adafruit_ads1x15.analog_in import AnalogIn  # single-ended channel view

            i2c = busio.I2C(board.SCL, board.SDA)           # open /dev/i2c-1
            ads = ADS.ADS1115(i2c, address=cfg.get("adc_address", 0x48))  # ADC chip
            pin = (ADS.P0, ADS.P1, ADS.P2, ADS.P3)[int(cfg.get("channel", 0)) % 4]
            self.chan = AnalogIn(ads, pin)                  # the probe's channel
            _log("moisture", f"SEN0193 on ADS1115 A{int(cfg.get('channel', 0)) % 4} ready")
        except Exception as exc:                            # ADC missing, address clash, etc.
            _log("moisture", f"unavailable: {exc}")         # moisture is logged as a dash

    def read(self):
        """Substrate moisture 0-100 %, or None if the probe is missing."""
        if self.chan is None:                               # no ADC channel
            return None                                     # caller shows a dash
        now = time.monotonic()                              # rate-limit clock
        if now - self._last_poll < 1.0:                     # once a second is plenty
            return self.last                                # cached percentage
        self._last_poll = now                               # this attempt counts
        try:
            volts = float(self.chan.voltage)                # probe output, 0-3.3 V
            span = max(self.dry - self.wet, 0.05)           # avoid dividing by zero
            # Dry reads high, wet reads low, so invert the ramp and clamp to 0-100.
            self.last = round(max(0.0, min(1.0, (self.dry - volts) / span)) * 100.0)
        except Exception:                                   # I2C glitch
            pass                                            # keep the previous value
        return self.last                                    # may be None if never read


class Battery:
    """INA219 on the 12 V rail."""

    def __init__(self):
        self.ina = None                                     # handle, if it opened
        try:
            import board                                    # Blinka pin names
            import busio                                    # Blinka I2C bus
            from adafruit_ina219 import INA219              # driver

            i2c = busio.I2C(board.SCL, board.SDA)           # open /dev/i2c-1
            self.ina = INA219(i2c, addr=0x40)               # address set by the breakout
            _log("batt", "INA219 ready")                    # boot log
        except Exception as exc:                            # not wired
            _log("batt", f"unavailable: {exc}")             # low-battery failsafe is disabled

    def volts(self):
        """Pack voltage in volts, or None."""
        if not self.ina:                                    # no monitor
            return None                                     # caller skips the failsafe
        try:
            return self.ina.bus_voltage + self.ina.shunt_voltage / 1000.0  # bus + shunt, mV->V
        except Exception:                                   # I2C glitch
            return None                                     # never crash a control tick


class SensorHub:
    """Creates every sensor once and exposes a single status snapshot."""

    def __init__(self, cfg):
        self.gnss = Gnss(cfg["gps"])                        # GPS, background thread
        self.imu = Imu(cfg["imu"])                          # compass / tilt
        self.thermal = Thermal(cfg["thermal"], cfg["thermal"].get("refresh_hz", 8))  # IR array
        self.lidar = Lidar(cfg["lidar"])                    # obstacle distance
        self.co2 = Co2(cfg.get("scd41", {}))                # CO2 ppm for detection
        self.temp = Temperature(cfg.get("ds18b20", {}))     # reference temperature
        self.moisture = Moisture(cfg.get("moisture", {}))   # sand moisture percent
        self.battery = Battery()                            # pack voltage

    def status(self, lidar_m=None):
        """Flat dict for the web API / CSV log.

        Pass a distance you have already read this tick to avoid reading the LiDAR
        twice — its serial read blocks for up to 250 ms. The three environment sensors
        cache internally, so calling this at 10 Hz costs almost nothing."""
        co2 = self.co2.read()                               # ppm or None
        ref_t = self.temp.read()                            # reference °C or None
        moisture = self.moisture.read()                     # percent or None
        if lidar_m is None:                                 # caller did not supply one
            lidar_m = self.lidar.read_m()                   # so read it here
        return {
            "lat": (self.gnss.position() or (None, None))[0],   # latitude or None
            "lon": (self.gnss.position() or (None, None))[1],   # longitude or None
            "fix": self.gnss.quality_text(),                # "NO FIX" / "GPS" / "RTK-FIX"
            "sats": self.gnss.sats,                         # satellites in use
            "hdop": round(self.gnss.hdop, 2),               # precision, 2 decimals
            "heading": (round(self.imu.heading(), 1)        # one decimal ...
                        if self.imu.heading() is not None else None),   # ... or a dash
            "lidar_m": round(lidar_m or -1.0, 2),           # -1 means "no reading"
            "co2_ppm": co2,                                 # integer ppm or None
            "ref_temp_c": None if ref_t is None else round(ref_t, 1),        # one decimal
            "moisture_pct": None if moisture is None else int(moisture),     # whole percent
            "battery_v": (round(self.battery.volts(), 2)    # two decimals ...
                          if self.battery.volts() is not None else None),    # ... or a dash
        }

