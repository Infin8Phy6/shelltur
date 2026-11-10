"""Turns one confirmed detection into every alert channel the rover has.

READING THIS FILE
The control loop must never wait for a modem, a siren or a blinking LED. detection()
only drops the event into a queue; a daemon thread does the slow work. That keeps the
10 Hz loop exactly as fast whether an alert takes 20 ms or 20 s.

Four channels, all optional:
  - the phone webapp, which already polls /api/status (see web/templates/index.html),
  - a flash pattern on the status LED,
  - a short siren on a spare GPIO (12 V siren through a MOSFET or relay),
  - an SMS through hardware/gsm.py, rate-limited and confidence-filtered.
"""
import queue                        # detections from the control loop
import threading                    # the alert worker runs on its own thread
import time                         # siren timing and the SMS cooldown

from gpiozero import OutputDevice   # a plain on/off pin for the siren


class Alerts:
    """Fan-out for detection events. Never blocks the caller."""

    def __init__(self, cfg, drive, gsm):
        self.cfg = cfg                                    # tunables from config.yaml
        self.drive = drive                                # owns the LED and buzzer
        self.gsm = gsm                                    # optional SMS sender
        self.flashes = int(cfg.get("led_flashes", 6))     # LED blink cycles
        self.siren_s = float(cfg.get("siren_s", 1.0))     # seconds of siren
        self.sms_min = float(cfg.get("sms_min_confidence", 0.6))
        self.sms_cooldown = float(cfg.get("sms_cooldown_s", 300))
        self._last_sms = 0.0                              # monotonic time of last text
        self.siren = None                                 # OutputDevice, if wired
        if cfg.get("siren", False):                       # only when the hardware exists
            try:
                self.siren = OutputDevice(int(cfg.get("siren_pin", 27)),
                                          active_high=True, initial_value=False)
                print("[alerts] siren ready", flush=True)
            except Exception as exc:                      # bad pin, no permission, etc.
                print(f"[alerts] siren unavailable: {exc}", flush=True)
        self.queue = queue.Queue(maxsize=4)               # a backlog of 4 is plenty
        threading.Thread(target=self._run, daemon=True).start()

    def detection(self, det):
        """Called from the control loop. Returns immediately."""
        try:
            self.queue.put_nowait(det)                    # hand it to the worker
        except queue.Full:                                # worker stuck on a modem
            pass                                          # drop it; the log still has it

    def _run(self):
        """Worker loop: pattern first (local), then SMS (slow)."""
        while True:
            det = self.queue.get()                        # next confirmed hit
            self._pattern()                               # LED, buzzer, siren
            self._sms(det)                                # maybe one text

    def _pattern(self):
        """Fast local alert: blink the LED, beep three times, wail briefly."""
        try:
            self.drive.led.blink(on_time=0.08, off_time=0.08,
                                 n=self.flashes, background=False)   # blocks ~1 s here
            self.drive.led.on()                           # back to steady "ready"
        except Exception:
            pass                                          # never let an LED kill an alert
        self.drive.beep(3, 0.08)                          # unmistakable triple beep
        if self.siren is not None:                        # only if wired and enabled
            try:
                self.siren.on()                           # 12 V siren on
                time.sleep(self.siren_s)                  # held for the configured time
                self.siren.off()                          # off again
            except Exception:
                pass

    def _sms(self, det):
        """Send one text if this hit is strong enough and the cooldown has passed."""
        if self.gsm is None or not getattr(self.gsm, "ready", False):
            return                                        # no modem, no text
        if det.confidence < self.sms_min:                 # weak hit: phone only
            return
        now = time.monotonic()                            # one clock for the cooldown
        if now - self._last_sms < self.sms_cooldown:      # too soon after the last one
            return
        self._last_sms = now                              # reserve this slot
        delta = "--" if det.delta_c is None else f"{det.delta_c:+.1f}"
        co2 = "--" if det.co2_ppm is None else det.co2_ppm
        text = (f"NestScout {det.kind} conf={det.confidence:.2f} "
                f"at {det.lat:.6f},{det.lon:.6f} "
                f"dT={delta}C CO2={co2}ppm")
        self.gsm.send(text)                               # queued, never blocking

    def close(self):
        """Switch the siren off and release its pin (called on shutdown)."""
        if self.siren is not None:                        # only if one was created
            try:
                self.siren.off()                          # make sure it is silent
                self.siren.close()                        # release the GPIO line
            except Exception:
                pass                                      # shutdown must never raise
