"""Skid-steer drive over two BTS7960 channels, with encoder odometry and a latched
emergency stop.

READING THIS FILE
"duty" is the motor power as a fraction: +1.0 = full speed forward, -1.0 = full speed
backward, 0 = stopped. The BTS7960 driver takes two PWM pins per motor, so gpiozero's
Motor class fits it directly.

Speed is open-loop duty (the beach is too loose for reliable wheel-speed control);
encoders are used for distance and stall detection, which is what the mission actually
needs.
"""
import math                          # for pi, when converting wheel turns to metres
import time                          # for monotonic clocks and the buzzer pauses

from gpiozero import Button, Buzzer, Motor, PWMLED, RotaryEncoder  # Pi pin helpers


def _approach(current, target, step):
    """Move `current` toward `target` by at most `step` — used to ramp the motors."""
    if target > current:                        # target is above us ...
        return min(target, current + step)      # ... so rise, but never overshoot
    return max(target, current - step)          # target is below us: fall, never undershoot


class Drive:
    """Owns the two motors, both encoders, the e-stop, buzzer and status LED."""

    def __init__(self, cfg):
        p = cfg["pins"]                          # short alias for the pin table
        self.cfg = cfg["robot"]                  # wheel size, ramp rate, speed limits
        self.ramp = float(self.cfg.get("ramp_per_s", 0.8))   # duty change per second

        # One gpiozero Motor per side. gpiozero puts PWM on both pins and accepts
        # -1..+1 on .value, which matches the BTS7960 RPWM/LPWM inputs.
        self.left = Motor(p["left"]["rpwm"], p["left"]["lpwm"])
        self.right = Motor(p["right"]["rpwm"], p["right"]["lpwm"])
        self._linv = -1.0 if p["left"].get("invert") else 1.0    # flip a mirrored gearbox
        self._rinv = -1.0 if p["right"].get("invert") else 1.0   # flip a mirrored gearbox

        # Quadrature encoders. max_steps=0 means "never wrap", so .steps is a running count.
        self.enc_left = RotaryEncoder(p["left"]["enc_a"], p["left"]["enc_b"], max_steps=0)
        self.enc_right = RotaryEncoder(p["right"]["enc_a"], p["right"]["enc_b"], max_steps=0)
        self._last_steps = (0, 0)                # encoder values at the previous speed sample
        self._last_dist_time = time.monotonic()  # timestamp of that sample
        self._speed = 0.0                        # last measured speed, m/s

        # E-stop: signal contact wired NO to GND, so pressed == stop.
        self.estop = Button(p["estop"], pull_up=True, bounce_time=0.05)  # 50 ms debounce
        self.buzzer = Buzzer(p["buzzer"])        # audible feedback for the operator
        self.led = PWMLED(p["led"])              # steady = ready, blinking = fault
        self.led.on()                            # show "ready" at boot

        self._l = 0.0                            # current ramp position, left motor
        self._r = 0.0                            # current ramp position, right motor
        self._goal = (0.0, 0.0)                  # where the ramp is heading
        self._last_tick = time.monotonic()       # timestamp of the previous tick()
        self.fault = None                        # "estop" once the e-stop has latched

    # ---------------------------------------------------------------- state
    @property
    def estop_active(self):
        """True while the e-stop button is physically pressed."""
        return self.estop.is_pressed             # pull-up input, button shorts to GND

    @property
    def stopped(self):
        """True if a fault is latched or the button is down."""
        return self.fault is not None or self.estop_active   # either condition blocks motion

    def reset_estop(self):
        """Clear the latched fault. Refuses while the button is still pressed."""
        if self.estop_active:                    # safety: no reset while still pressed
            return False                         # tell the caller it did not work
        self.fault = None                        # clear the latch
        self.led.on()                            # back to steady = ready
        return True                              # reset accepted

    # ----------------------------------------------------------------- drive
    def command(self, left, right):
        """Set duty targets in -1..+1 (positive drives forward)."""
        self._goal = (max(-1.0, min(1.0, left)), max(-1.0, min(1.0, right)))  # clamp both

    def arcade(self, throttle, steer):
        """throttle/steer in -1..+1; steer positive turns right."""
        left = throttle + steer                  # right turn adds to the left wheel
        right = throttle - steer                 # ... and subtracts from the right wheel
        peak = max(abs(left), abs(right), 1.0)   # 1.0 means "no scaling needed"
        self.command(left / peak, right / peak)  # scale down only if a wheel exceeds 1

    def stop(self):
        """Stop now, bypassing the ramp. Used by aborts and failsafes."""
        self._goal = (0.0, 0.0)                  # target zero for the ramp as well
        self._l = self._r = 0.0                  # snap the ramp position to zero
        self._write(0.0, 0.0)                    # push zero duty to the drivers

    def tick(self):
        """Call at 20-50 Hz. Applies ramping and the e-stop interlock."""
        now = time.monotonic()                   # current time
        dt = min(now - self._last_tick, 0.25)    # seconds since last tick, capped after a stall
        self._last_tick = now                    # remember for the next call

        if self.estop_active:                    # button held down?
            if self.fault != "estop":            # only blink once, not every tick
                self.fault = "estop"             # latch the fault
                self.led.blink(on_time=0.1, off_time=0.1)  # fast blink = stopped
        if self.stopped:                         # fault or button ...
            self._l = self._r = 0.0              # ... force the ramp to zero
            self._write(0.0, 0.0)                # ... and cut the motors
            return                               # nothing else to do

        step = self.ramp * dt                    # how much duty may change this tick
        self._l = _approach(self._l, self._goal[0], step)   # ease left toward its target
        self._r = _approach(self._r, self._goal[1], step)   # ease right toward its target
        self._write(self._l, self._r)            # send the ramped values to the drivers

    def _write(self, left, right):
        """Apply the invert flags and push duty to both motors."""
        self.left.value = left * self._linv      # -1..+1, flipped if needed
        self.right.value = right * self._rinv    # -1..+1, flipped if needed

    # ------------------------------------------------------------- odometry
    def _metres_per_step(self):
        """Distance travelled per encoder step (average of both wheels)."""
        return math.pi * float(self.cfg["wheel_diameter"]) / float(self.cfg["ticks_per_rev"])

    def distance_m(self):
        """Distance since boot, averaged over the two wheels."""
        steps = (self.enc_left.steps + self.enc_right.steps) / 2.0   # mean of both encoders
        return steps * self._metres_per_step()                       # convert to metres

    def reset_odometry(self):
        """Zero the distance counters (e.g. before a measured test run)."""
        self.enc_left.steps = 0                  # left counter back to zero
        self.enc_right.steps = 0                 # right counter back to zero
        self._last_steps = (0, 0)                # forget the old speed sample

    def speed_mps(self):
        """Measured speed in m/s, updated at most five times per second."""
        now = time.monotonic()                   # timestamp for this sample
        steps = (self.enc_left.steps, self.enc_right.steps)  # both counters right now
        dt = now - self._last_dist_time          # time since the previous sample
        if dt >= 0.2:                            # only recalculate 5 times a second
            delta = ((steps[0] - self._last_steps[0]) + (steps[1] - self._last_steps[1])) / 2.0
            self._speed = delta * self._metres_per_step() / dt   # metres / seconds
            self._last_steps = steps             # remember for next time
            self._last_dist_time = now           # remember for next time
        return self._speed                       # last known speed

    def stalled(self, threshold_mps=0.02):
        """True when we are commanded to move but the wheels are not turning."""
        moving_cmd = abs(self._goal[0]) > 0.15   # is the left motor being told to move?
        return moving_cmd and abs(self.speed_mps()) < threshold_mps   # command but no motion

    # ---------------------------------------------------------------- misc
    def beep(self, count=1, duration=0.12):
        """Short beeps for operator feedback (home set, detection, abort)."""
        for _ in range(count):                   # repeat the beep
            self.buzzer.on()                     # sound on ...
            time.sleep(duration)                 # ... for the requested time ...
            self.buzzer.off()                    # ... sound off ...
            time.sleep(0.08)                     # ... and a gap before the next beep

    def close(self):
        """Stop the motors and release every GPIO line (called on shutdown)."""
        self.stop()                              # never leave the motors driven
        for device in (self.left, self.right, self.enc_left, self.enc_right,
                       self.buzzer, self.led):   # every gpiozero object we own
            try:
                device.close()                   # release the pin
            except Exception:
                pass                             # shutdown must never raise
