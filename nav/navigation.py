"""Waypoint following and the scan/home state machine.

READING THIS FILE
Pilot        = "drive toward this GPS point" (one point at a time).
MissionRunner = "which point is next, and when are we finished".

The split matters: the pilot knows nothing about surveys, and the mission knows nothing
about motor duties. Each is easy to test on its own.
"""
import time                          # for the settle timer and mission runtime

from nav.geo import angle_diff, bearing_deg, distance_m   # position maths from geo.py


class Pilot:
    """Heading-hold waypoint driver for a skid-steer chassis.

    Tuning: raise kp until the rover tracks a lane without weaving; add kd only if it
    overshoots. On loose sand a little weave is normal — do not chase it.
    """

    def __init__(self, drive, imu, robot_cfg, mission_cfg):
        self.drive = drive                                   # motor layer
        self.imu = imu                                       # provides the compass heading
        self.robot = robot_cfg                               # speed limits
        self.arrive_radius = mission_cfg["arrive_radius_m"]  # how close counts as "there"
        self.kp = 0.030                                      # proportional steering gain
        self.kd = 0.004                                      # derivative (damping) gain
        self.target = None                                   # the point we are driving to
        self._last_error = 0.0                               # previous heading error, for kd
        self.last_result = "idle"                            # human-readable status

    def set_target(self, point):
        """Start driving to a new (lat, lon) point."""
        self.target = point                                  # store the destination
        self._last_error = 0.0                               # forget the old derivative

    def clear(self):
        """Forget the target and stop."""
        self.target = None                                   # no destination
        self.drive.stop()                                    # motors off now

    def update(self, position):
        """Call every control tick. Returns idle / no_target / no_heading / arrived / driving."""
        if self.target is None or position is None:          # nothing to do, or no GPS fix
            self.drive.stop()                                # make sure we are stopped
            self.last_result = "no_target"                   # report why
            return self.last_result                          # leave early

        remaining = distance_m(position, self.target)        # metres still to travel
        if remaining <= self.arrive_radius:                  # close enough?
            self.drive.stop()                                # brake
            self.target = None                               # this waypoint is done
            self.last_result = "arrived"                     # tell the mission to advance
            return self.last_result                          # leave early

        heading = self.imu.heading()                         # where the rover points now
        if heading is None:                                  # compass unavailable
            self.drive.stop()                                # never drive blind
            self.last_result = "no_heading"                  # report it
            return self.last_result                          # leave early

        error = angle_diff(bearing_deg(position, self.target), heading)  # signed turn needed
        derivative = error - self._last_error                # how fast the error is changing
        self._last_error = error                             # remember for the next tick

        steer = max(-1.0, min(1.0, self.kp * error + self.kd * derivative))  # PD -> duty
        speed_factor = 1.0 if remaining > 2.5 else max(remaining / 2.5, 0.35)  # slow near target
        if abs(error) > 60.0:                                # badly misaligned ...
            speed_factor *= 0.3          # pivot first, then drive
        # Open-loop duty: max_speed/manual_speed keeps the crawl inside the joystick range.
        throttle = (self.robot["max_speed"] / max(self.robot["manual_speed"], 1e-3)) * speed_factor

        self.drive.arcade(throttle * 0.9, steer * 0.65)      # 0.9 and 0.65 leave headroom
        self.last_result = "driving"                         # report progress
        return self.last_result                              # done for this tick


class MissionRunner:
    """SCAN = drive the grid. RETURN_HOME = drive to home and stop. IDLE = manual."""

    def __init__(self, pilot, cfg):
        self.cfg = cfg                                       # mission settings
        self.pilot = pilot                                   # the waypoint driver
        self.mode = "idle"                                   # current state
        self.waypoints = []                                  # points still to visit
        self.index = 0                                       # which point is next
        self.home = None                                     # the recorded home coordinate
        self.started = 0.0                                   # start time, for runtime limits
        self._settle_until = 0.0                             # pause at a waypoint until then

    # ------------------------------------------------------------- control
    def start_scan(self, waypoints, home):
        """Begin a survey along a pre-generated list of points."""
        if not waypoints:                                    # nothing to drive
            return False                                     # refuse the mission
        self.waypoints = list(waypoints)                     # copy, so the caller cannot mutate it
        self.index = 0                                       # start at the first point
        self.home = home                                     # remember where home is
        self.mode = "scan"                                   # enter SCAN state
        self.started = time.monotonic()                      # start the runtime clock
        self._settle_until = 0.0                             # no pause before the first move
        self.pilot.set_target(self.waypoints[0])             # start driving
        return True                                          # mission accepted

    def return_home(self, home):
        """Drive to a home point and stop there."""
        if home is None:                                     # we have nowhere to go
            return False                                     # refuse
        self.home = home                                     # store it
        self.waypoints = [home]                              # a one-point mission
        self.index = 0                                       # that point is next
        self.mode = "return_home"                            # enter RETURN_HOME state
        self.started = time.monotonic()                      # start the runtime clock
        self._settle_until = 0.0                             # move immediately
        self.pilot.set_target(home)                          # start driving
        return True                                          # accepted

    def abort(self):
        """Stop everything and go idle (operator pressed STOP/ABORT)."""
        self.mode = "idle"                                   # leave the mission states
        self.waypoints = []                                  # forget the grid
        self.index = 0                                       # reset the counter
        self.pilot.clear()                                   # stop the motors

    # ---------------------------------------------------------------- tick
    def tick(self, position):
        """Advance the mission one control tick. Called at 10 Hz."""
        if self.mode not in ("scan", "return_home"):         # only these modes drive themselves
            return                                           # manual/idle: nothing to do
        if position is None:                                 # no GPS fix
            self.pilot.drive.stop()                          # hold position
            return                                           # retry next tick
        if time.monotonic() < self._settle_until:            # still paused at a waypoint?
            self.pilot.drive.stop()                          # stay still
            return                                           # let the thermal frame settle

        if self.pilot.update(position) != "arrived":         # drive toward the current point
            return                                           # not there yet

        if self.mode == "return_home":                       # we were only going home
            self.mode = "idle"                               # mission complete
            self.pilot.drive.beep(2)                         # audible "I am home"
            return                                           # done

        self.index += 1                                      # move to the next waypoint
        if self.index >= len(self.waypoints):                # was that the last one?
            self.mode = "idle"                               # survey finished
            self.pilot.drive.beep(3)                         # audible "finished"
            return                                           # done

        self.pilot.set_target(self.waypoints[self.index])    # aim at the next point
        self._settle_until = time.monotonic() + self.cfg["settle_s"]  # brief pause to sample

    @property
    def progress(self):
        """Percent of the grid completed, for the phone UI."""
        if not self.waypoints:                               # no mission
            return 0.0                                       # avoid divide-by-zero
        return round(100.0 * self.index / len(self.waypoints), 1)   # one decimal place

    @property
    def settled(self):
        """True when the settle pause is over (detector may resume sampling)."""
        return time.monotonic() >= self._settle_until        # compare clocks

    @property
    def elapsed_min(self):
        """Minutes since the mission started, for the runtime limit."""
        if not self.started:                                 # no mission yet
            return 0.0                                       # nothing elapsed
        return (time.monotonic() - self.started) / 60.0      # seconds -> minutes
