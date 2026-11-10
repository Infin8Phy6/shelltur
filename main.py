#!/usr/bin/env python3
"""NestScout controller.

One control loop at 10 Hz owns the hardware. The Flask app never touches GPIO; it only
posts commands into a queue and reads the telemetry snapshot, so a slow phone or a dead
Wi-Fi link cannot stall the rover."""
# `#!/usr/bin/env python3` lets you run the file directly (`./main.py`) on Linux.
# The docstring explains the architecture: exactly one thread (below) talks to the
# motors, camera and sensors. The web thread only drops messages in a queue and
# copies numbers out, so a hung browser cannot freeze the rover.
import queue        # bounded FIFO used to pass commands from the web thread to the loop
import threading    # Lock (protects telemetry) and Thread (starts the control loop)
import time         # monotonic clock (loop pacing, timeouts) and strftime for CSV stamps
from pathlib import Path  # portable file-system paths (`ROOT / "data"`)
import yaml         # reads config.yaml into a plain dict

from detect.nest_detector import NestDetector   # thermal frame -> confirmed nest detection
from hardware.alerts import Alerts              # LED / siren / SMS alert fan-out
from hardware.camera import Camera              # picamera2 wrapper (MJPEG + JPEG stills)
from hardware.drive import Drive                # motors, encoders, buzzer, e-stop
from hardware.gsm import Gsm                    # optional GSM modem for SMS alerts
from hardware.sensors import SensorHub          # GNSS, IMU, LiDAR, CO2, temp, moisture, battery
from nav.geo import distance_m, scan_grid       # haversine distance, lawnmower pattern
from nav.navigation import MissionRunner, Pilot  # mission state machine, PD steering
from web.app import create_app                   # factory that builds the Flask app

ROOT = Path(__file__).resolve().parent  # directory containing this file: ~/nestscout
LOOP_HZ = 10.0                          # control loop rate: 10 ticks per second (100 ms)


class Robot:
    def __init__(self, cfg):
        # `cfg` is the parsed config.yaml dict. Everything below is built once, here,
        # so that a typo in the YAML fails loudly at start-up instead of mid-mission.
        self.cfg = cfg                    # keep the whole config for later reads (failsafes)
        (ROOT / "data").mkdir(exist_ok=True)  # make sure the log folder exists before we append

        self.drive = Drive(cfg)                    # motors + encoders; starts with outputs low
        self.gsm = Gsm(cfg.get("gsm", {}))          # optional SMS modem (queue + thread)
        self.alerts = Alerts(cfg.get("alerts", {}), self.drive, self.gsm)   # alert fan-out
        self.sensors = SensorHub(cfg)               # opens serial / I2C devices (may raise)
        self.camera = Camera(cfg["camera"])         # opens the CSI camera (may be disabled)
        # The detector needs sensors + camera because it reads the IMU heading and saves photos.
        self.detector = NestDetector(cfg["detector"], self.sensors, self.camera, ROOT / "data")
        # Pilot converts "aim at this point" into throttle/steer; it owns the motors' goals.
        self.pilot = Pilot(self.drive, self.sensors.imu, cfg["robot"], cfg["mission"])
        # MissionRunner sequences waypoints and tells the Pilot where to go next.
        self.mission = MissionRunner(self.pilot, cfg["mission"])

        # maxsize=32 is deliberate: if the web thread floods us we drop commands rather
        # than grow without bound and starve the control loop of time.
        self.commands = queue.Queue(maxsize=32)
        self.lock = threading.Lock()   # guards `telemetry` between the loop and web threads
        self.home = None               # (lat, lon) of the arming spot; set by arm_here/scan
        self.detections = []           # every confirmed nest this session (for the UI log)
        self.last_heartbeat = time.monotonic()  # last joystick command; timeout -> stop
        self.no_fix_since = None       # timestamp when GNSS lost its fix, or None
        self.arming = False            # True while averaging fixes to set "home"
        self._arm_samples = []         # list of (lat, lon) collected during arming
        self._arm_deadline = 0.0       # monotonic time when arming finishes
        self._track_last = 0.0         # last time the track CSV was written (throttle to 1 Hz)
        self._detection_file = ROOT / "data" / "detections.csv"  # one CSV for all detections
        self.telemetry = {"mode": "idle"}  # snapshot dict; replaced wholesale each tick

    # ------------------------------------------------------------ commands
    def send(self, command):
        # Called from the Flask thread. Never blocks: a full queue means the loop is
        # busy, and dropping the oldest-ish command is better than stalling HTTP.
        try:
            self.commands.put_nowait(command)   # add without waiting
        except queue.Full:
            pass                                # silently ignore; next command will fit

    def handle(self, command):
        # Runs inside the control loop (single thread => no locks needed here).
        name = command.get("cmd")           # every command is {"cmd": "...", ...}
        if name == "drive":
            # Joystick input. Refuse while a GPS mission owns the motors.
            if self.mission.mode in ("scan", "return_home"):
                return                      # autonomous mode wins; ignore manual drive
            self.mission.mode = "manual"    # manual takeover cancels any idle state
            self.last_heartbeat = time.monotonic()   # feed the 1 s watchdog
            # Throttle/steer are -1..1 floats; arcade() mixes them into left/right goals.
            self.drive.arcade(float(command.get("throttle", 0)),
                              float(command.get("steer", 0)))
        elif name == "stop":
            self.mission.abort()            # drop back to idle and forget the waypoint list
            self.drive.stop()               # zero both motor goals immediately
        elif name == "arm_here":
            # "This spot is home." Average GNSS fixes for `home_settle_s` seconds.
            if self.sensors.gnss.fix_ok:    # refuse to arm without a usable fix
                self.arming = True          # the main loop collects samples while this is True
                self._arm_samples = []      # start a fresh average
                # Honour an explicit `seconds` from the phone, else use config (10 s).
                self._arm_deadline = time.monotonic() + float(
                    command.get("seconds") or self.cfg["gps"]["home_settle_s"])
        elif name == "return_home":
            # Drive back to the remembered home point (only if we have one).
            if self.home and self.mission.return_home(self.home):
                self.drive.beep(1)          # one short beep = accepted
        elif name == "goto":
            # Same as return_home but with a phone-supplied target (geolocation button).
            target = (float(command["lat"]), float(command["lon"]))
            if self.mission.return_home(target):   # reuses the single-waypoint state machine
                self.drive.beep(1)
        elif name == "scan":
            # Full survey: build a lawnmower grid between the two map corners.
            p1 = (float(command["lat1"]), float(command["lon1"]))  # first corner
            p2 = (float(command["lat2"]), float(command["lon2"]))  # opposite corner
            lanes = int(command.get("lanes", 1))                   # passes across the beach
            # Phone may override lane spacing; otherwise take it from config (1.0 m).
            spacing = float(command.get("spacing")
                            or self.cfg["mission"]["lane_spacing_m"])
            # scan_grid returns waypoints spaced `sample_spacing_m` apart along each lane.
            grid = scan_grid(p1, p2, lanes, spacing,
                             self.cfg["mission"]["sample_spacing_m"])
            if self.home is None:
                # No arming yet: average 5 fixes right now so we still have a way back.
                self.home = self.sensors.gnss.average_position(5)
            if self.mission.start_scan(grid, self.home):   # False if the grid is empty
                self.drive.beep(1)          # one beep = survey started
        elif name == "abort":
            # Panic button on the phone: stop everything and announce it.
            self.mission.abort()
            self.drive.stop()
            self.drive.beep(2)              # two beeps = aborted
        elif name == "estop_reset":
            # Re-arm the drive after the e-stop button was released.
            if self.drive.reset_estop():    # False while the physical button is still down
                self.drive.beep(1)

    # -------------------------------------------------------------- helpers
    def _commit_home(self):
        # Turn the collected arming samples into one home coordinate (the mean).
        samples = self._arm_samples
        # Averaging lat/lon separately is fine over a few metres on a beach.
        self.home = (sum(p[0] for p in samples) / len(samples),
                     sum(p[1] for p in samples) / len(samples))
        self.arming = False             # stop collecting samples in the main loop
        self.drive.beep(2)              # two beeps = home stored (audible without the phone)
        # 7 decimal places is about 1 cm — more than the GNSS itself can resolve.
        print(f"[home] set to {self.home[0]:.7f}, {self.home[1]:.7f}", flush=True)

    def _log_detection(self, det):
        # Append one detection to data/detections.csv; write a header if the file is new.
        row = det.as_dict()             # Detection -> plain dict of column values
        new = not self._detection_file.exists()   # True only on the very first detection
        # `with open(...,"a")` appends, so we never lose earlier rows on a crash.
        with open(self._detection_file, "a", encoding="utf-8") as handle:
            if new:
                # Column order here must match the join below, in the same order.
                handle.write("time,lat,lon,z,temp_c,delta_c,co2_ppm,moisture_pct,"
                             "confidence,cnn_prob,consistency,kind,frames,photo\n")
            # Join the values in a fixed order; str() handles floats and text alike.
            handle.write(",".join(str(row[k]) for k in
                                  ("time", "lat", "lon", "z", "temp_c", "delta_c",
                                   "co2_ppm", "moisture_pct", "confidence", "cnn_prob",
                                   "consistency", "kind", "frames", "photo")) + "\n")

    def _log_track(self, position):
        # Record where the rover was, at most once per second, in a per-day CSV.
        if position is None or time.monotonic() - self._track_last < 1.0:
            return                      # no fix yet, or we logged less than 1 s ago
        self._track_last = time.monotonic()   # remember when we last wrote
        # One file per day keeps each file small and easy to plot later.
        path = ROOT / "data" / "tracks" / time.strftime("track-%Y%m%d.csv")
        new = not path.exists()         # a fresh file needs a header row
        with open(path, "a", encoding="utf-8") as handle:
            if new:
                handle.write("time,lat,lon,mode\n")
            # Mode tells you whether a point was autonomous ("scan") or joystick ("manual").
            handle.write(f"{time.strftime('%H:%M:%S')},{position[0]:.7f},"
                         f"{position[1]:.7f},{self.mission.mode}\n")

    def _failsafe(self, position, clearance):
        """Returns True if the rover was forced to stop this tick."""
        # Highest priority: something is directly in front while driving autonomously.
        stop_m = self.cfg["lidar"]["obstacle_stop_m"]   # 0.90 m from config.yaml
        if (clearance is not None and clearance < stop_m
                and self.mission.mode in ("scan", "return_home")):
            self.drive.stop()           # brake now; the loop will not tick the mission
            return True

        # Second: not enough battery to finish the survey. Limp home instead.
        low = self.cfg.get("battery", {}).get("low_v", 12.1)  # 12.1 V default
        volts = self.sensors.battery.volts()
        # The check is skipped in return_home so we keep trying to get back even if low.
        if volts is not None and volts < low and self.mission.mode in ("scan", "manual"):
            print(f"[batt] {volts:.2f} V — returning home", flush=True)
            self.mission.abort()        # throw away the remaining waypoints
            self.drive.stop()
            if self.home:
                self.mission.return_home(self.home)   # then drive to the arming spot
            return True

        # Third: the phone went quiet. Joystick mode needs a command every second.
        if self.mission.mode in ("manual",) and \
                time.monotonic() - self.last_heartbeat > self.cfg["web"]["command_timeout_s"]:
            self.drive.stop()           # dead-man switch: no heartbeat, no motion
            return True

        # GNSS problems only matter for autonomous driving.
        if self.mission.mode in ("scan", "return_home"):
            if not self.sensors.gnss.fix_ok:
                self.drive.stop()       # never drive blind on a stale position
                if self.no_fix_since is None:
                    self.no_fix_since = time.monotonic()      # start the grace period
                elif time.monotonic() - self.no_fix_since > 20:  # 20 s without a fix
                    self.mission.abort()  # give up on the mission
                    print("[gnss] fix lost for 20 s — mission aborted", flush=True)
                return True
            self.no_fix_since = None    # fix came back: reset the grace timer

            # Geofence: if the survey dragged us too far from home, drive back.
            if (self.home and self.mission.mode == "scan"
                    and distance_m(position, self.home)
                    > self.cfg["mission"]["max_home_distance_m"]):   # 400 m
                print("[geofence] too far from home — returning", flush=True)
                self.mission.abort()
                self.mission.return_home(self.home)
                return True

            # Runtime cap: a 90 min limit stops a mis-planned survey from running all day.
            if self.mission.elapsed_min > self.cfg["mission"]["max_runtime_min"]:
                print("[mission] runtime limit — returning", flush=True)
                self.mission.abort()
                self.mission.return_home(self.home)
                return True
        return False                    # nothing tripped: the mission may tick normally

    def snapshot(self):
        # Called by the Flask thread for /api/status. Copy under the lock, then release
        # it before serialising so a slow JSON dump cannot delay the control loop.
        with self.lock:
            state = dict(self.telemetry)   # shallow copy of the latest numbers
            # Only the last 20 detections are sent: enough for the phone list, small payload.
            state["detections"] = [d.as_dict() for d in self.detections[-20:]]
        # These four are cheap single reads and live outside the lock on purpose.
        state["home"] = self.home
        state["estop"] = self.drive.estop_active   # drives the red banner in the UI
        state["fault"] = self.drive.fault          # "estop" or a motor-driver problem
        state["arming"] = self.arming              # shows the "averaging" spinner
        return state

    # ----------------------------------------------------------- main loop
    def run(self):
        """Supervisor: a sensor exception must never leave the motors latched."""
        # If any sensor throws (unplugged LiDAR, USB hiccup), we catch it here, stop the
        # motors, wait half a second and restart the loop instead of dying silently.
        while True:
            try:
                self._run_loop()        # normally runs forever
            except Exception as exc:
                print(f"[loop] crashed: {exc}", flush=True)
                self.drive.stop()       # fail safe: never leave a nonzero motor goal
                time.sleep(0.5)         # avoid a hot spin if the fault is permanent

    def _run_loop(self):
        period = 1.0 / LOOP_HZ         # 0.1 s: the nominal length of one tick
        while True:
            started = time.monotonic()  # timestamp for the pacing calculation at the end
            self.drive.tick()           # apply ramping, read encoders, check e-stop/stall

            # Drain the whole command queue this tick (it is capped at 32 entries).
            while not self.commands.empty():
                try:
                    self.handle(self.commands.get_nowait())
                except queue.Empty:
                    break               # another consumer raced us; queue is now empty

            # Read the three sensors the loop itself needs. LiDAR is read once here and
            # its value is passed to status() so we do not block on the serial port twice.
            position = self.sensors.gnss.position()   # (lat, lon) or None
            heading = self.sensors.imu.heading()      # degrees clockwise from north or None
            altitude = self.sensors.lidar.read_m()    # forward clearance in metres or None

            if self.arming:
                # Collect one fix per tick (10 per second) until the deadline passes.
                if self.sensors.gnss.fix_ok:
                    self._arm_samples.append(position)
                if time.monotonic() >= self._arm_deadline:
                    if self._arm_samples:
                        self._commit_home()       # average them and store "home"
                    else:
                        self.arming = False       # never got a fix; give up quietly
                        print("[home] no fixes during arming", flush=True)

            # Failsafes run before the mission, so a trip means the mission is skipped.
            forced_stop = self._failsafe(position, altitude)
            if not forced_stop:
                self.mission.tick(position)  # drive toward the next waypoint (if any)

            # Thermal detection only runs during a survey that is not failsafe-stopped.
            if self.mission.mode == "scan" and not forced_stop:
                detection = self.detector.update(position, heading)
                if detection:                # None unless `confirm_frames` agreed
                    self.detections.append(detection)   # keep it for /api/status
                    self._log_detection(detection)      # and for detections.csv
                    self.alerts.detection(detection)    # LED + siren + SMS, off the loop
                    print(f"[detect] {detection.kind} conf={detection.confidence:.2f} "
                          f"z={detection.zscore:.2f} "
                          f"at {detection.lat:.7f},{detection.lon:.7f}", flush=True)

            self._log_track(position)        # 1 Hz breadcrumb trail (no-op without a fix)

            status = self.sensors.status(altitude)   # GNSS/IMU/battery/environment dict
            with self.lock:
                # Build a brand-new dict and swap it in: readers never see a half update.
                self.telemetry = {
                    **status,                # everything from SensorHub.status()
                    "mode": self.mission.mode,
                    "progress": self.mission.progress,   # 0..1 for the progress bar
                    # "3/41" style label; safe when there are no waypoints.
                    "waypoint": f"{self.mission.index}/{len(self.mission.waypoints)}",
                    # Round for the UI: a long float is noise on a phone screen.
                    "heading": None if heading is None else round(heading, 1),
                    "lidar_m": None if altitude is None else round(altitude, 2),
                    "stalled": self.drive.stalled(),     # True if commanded but not moving
                    "detection_count": len(self.detections),
                }

            # Sleep only the unused part of the period, so slow ticks do not drift
            # cumulatively. If the tick overran, sleep is negative and we just carry on.
            sleep = period - (time.monotonic() - started)
            if sleep > 0:
                time.sleep(sleep)


def load_config():
    # One place that knows where the config lives, so main() stays short.
    with open(ROOT / "config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)   # YAML -> dict of dicts (no Python objects)


def main():
    cfg = load_config()                 # parse config.yaml (raises if missing/broken)
    robot = Robot(cfg)                  # open every device; fail fast here, not mid-mission
    app = create_app(robot)             # Flask app; it only ever calls robot.send/snapshot
    # daemon=True means Ctrl-C on the main thread will not wait for the loop to finish.
    threading.Thread(target=robot.run, daemon=True).start()
    robot.drive.beep(1)                 # one beep = controller is alive and listening
    try:
        # use_reloader=False is required: the reloader would run main() twice and the
        # second copy would fight for the camera and GPIO pins.
        app.run(host=cfg["web"]["host"], port=cfg["web"]["port"],
                threaded=True, use_reloader=False)
    finally:
        # This runs on Ctrl-C or any exception in app.run: leave the hardware safe.
        robot.drive.stop()              # zero the motors
        robot.camera.close()            # release the CSI camera
        robot.alerts.close()            # silence the siren, release its pin
        robot.gsm.close()               # release the modem's USB port
        robot.drive.close()             # release GPIO pins and switch off the buzzer


if __name__ == "__main__":
    main()                              # only auto-run when executed, not when imported
