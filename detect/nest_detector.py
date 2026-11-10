"""Turns thermal frames into geotagged nest candidates.

READING THIS FILE
The thermal array gives 24x32 temperatures, but a nest is not a temperature — it is a
*pattern*: a small group of cells that stays warmer (or cooler) than the sand around it,
in the same physical place, for several frames.

So each frame goes through the six requested inputs plus a CNN pattern check, and every
confirmed nest carries all of them:

  1. Thermal camera reading — robust z-score of the frame (median / MAD), then the
     largest connected cluster of |z| > threshold on the warm and cool side.
  2. Carbon dioxide level — the SCD41 reading compared with the median of the last
     `co2_baseline_s` seconds, so a high background does not block everything.
  3. Sand temperature — the cluster temperature compared with the DS18B20 reference.
  4. Sand moisture — the SEN0193 reading must fall inside a plausible nesting band.
  5. Consistency of readings — the same spot must keep giving the same temperature and
     position across `confirm_frames` frames, not just appear once.
  6. Location at time — the pixel is projected through the mount geometry and heading
     into a lat/lon, stamped with the time, and scored by GNSS fix quality.
  7. Thermal CNN — a tiny model scores the normalised frame for a buried-shell pattern.
     It is what lets a +0.4..0.6 C anomaly through: when the CNN is confident the ΔT
     gate relaxes from `min_delta_c` to `weak_delta_c`, and the cluster threshold
     relaxes from `z_threshold` to `weak_z_threshold`.

The seven inputs are combined into a 0-1 confidence for ranking hits. CO2, ΔT, moisture
and consistency are gates — a frame we do not trust is dropped. Thermal strength, the
CNN and fix quality set the score. A sensor (or model) that is missing drops its input
instead of silencing the detector.
"""
import math                          # tan/atan for the camera projection
import time                          # timestamps on detections
from dataclasses import dataclass    # a tiny result record

import numpy as np                   # frame maths

from detect.thermal_cnn import ThermalCnn          # weak-contrast pattern classifier
from nav.geo import distance_m, ground_to_latlon   # GPS maths


def _clamp(value, low=0.0, high=1.0):
    """Keep a score inside 0..1 without importing anything."""
    return max(low, min(high, value))


@dataclass
class Detection:
    """One confirmed nest candidate, ready for the log and the phone UI."""
    lat: float                       # projected latitude of the candidate
    lon: float                       # projected longitude of the candidate
    zscore: float                    # mean anomaly strength (higher = clearer)
    temp_c: float                    # mean cluster temperature
    kind: str                        # "warm" or "cool" anomaly
    frames: int                      # how many frames agreed
    time: float                      # unix timestamp of confirmation
    photo: str = ""                  # path to the evidence JPEG, empty if unavailable
    co2_ppm: int = None              # CO2 at confirmation, None if no sensor
    delta_c: float = None            # cluster minus reference temperature, None if no probe
    moisture_pct: int = None         # sand moisture at confirmation, None if no probe
    confidence: float = 0.0          # 0-1 composite of the inputs, for ranking
    consistency: float = 0.0         # 0-1 repeatability of the last `confirm_frames` hits
    cnn_prob: float = None           # CNN shell probability, None if no model

    def as_dict(self):
        """JSON/CSV friendly version. Rounds for readability, keeps 7 lat/lon digits."""
        return {
            "lat": round(self.lat, 7), "lon": round(self.lon, 7),      # ~1 cm precision
            "z": round(self.zscore, 2), "temp_c": round(self.temp_c, 2),   # 2 decimals
            "kind": self.kind, "frames": self.frames,                  # plain values
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.time)),  # readable
            "photo": self.photo,                                       # file path
            "co2_ppm": self.co2_ppm,                                   # int or null
            "delta_c": None if self.delta_c is None else round(self.delta_c, 2),  # 2 decimals
            "moisture_pct": self.moisture_pct,                         # int or null
            "confidence": round(self.confidence, 2),                   # composite score
            "consistency": round(self.consistency, 2),                 # repeatability score
            "cnn_prob": None if self.cnn_prob is None else round(self.cnn_prob, 2),  # model score
        }


class _Group:
    """A running average of hits that belong to the same physical spot.

    This is what makes the detector stable: the rover moves, the cluster moves across
    the frame, but the projected GPS coordinate stays put, so the hits accumulate here.
    """

    def __init__(self, lat, lon, z, temp, kind, now):
        self.lat, self.lon, self.n = lat, lon, 1     # first hit, count = 1
        self.first, self.last = now, now             # first and latest hit time
        self.z_sum = z                               # first hit's strength, not zero
        self.temp_sum = temp                         # first hit's temperature, not zero
        self.kind = kind                             # warm or cool
        self.temps = [temp]                          # every temperature, for consistency
        self.points = [(lat, lon)]                   # every position, for consistency

    def add(self, lat, lon, z, temp, kind, now):
        """Merge one more hit into the running mean."""
        self.lat = (self.lat * self.n + lat) / (self.n + 1)        # incremental latitude mean
        self.lon = (self.lon * self.n + lon) / (self.n + 1)        # incremental longitude mean
        self.z_sum += z                              # accumulate strength
        self.temp_sum += temp                        # accumulate temperature
        self.kind = kind                             # the latest classification wins
        self.temps.append(temp)                      # remember it for the spread
        self.points.append((lat, lon))               # remember where it landed
        self.n += 1                                  # one more agreeing frame
        self.last = now                              # refresh the expiry clock

    def consistency(self, radius_m, temp_tolerance_c):
        """0-1: did the same spot keep giving the same answer?

        Two things must hold for a nest: the cluster temperature should not swing
        between frames, and the projected position should not wander. A passing cloud,
        a footstep or a hot rock tends to fail one of the two. CO2 is deliberately not
        included here — it only updates every 5 s, so per-frame spread would be noise."""
        if self.n < 2:                               # a single hit has no spread
            return 0.0                               # cannot claim consistency
        temp_std = float(np.std(self.temps))         # frame-to-frame temperature spread
        # The configured tolerance is the half-way point, so twice the tolerance scores 0.
        temp_ok = _clamp(1.0 - temp_std / max(2.0 * temp_tolerance_c, 0.1))
        centre = (self.lat, self.lon)                # running mean position
        spread = float(np.mean([distance_m(p, centre) for p in self.points]))   # metres
        pos_ok = _clamp(1.0 - spread / max(2.0 * radius_m, 0.2))         # 1 = same spot
        return (temp_ok + pos_ok) / 2.0              # both must be good to score high


class NestDetector:
    """Stateless maths plus a small amount of state (open groups, suppressions)."""

    def __init__(self, cfg, sensors, camera, data_dir):
        self.cfg = cfg                               # thresholds and mount geometry
        self.mount = cfg["mount"]                    # height, tilt, field of view
        self.sensors = sensors                       # thermal frame source
        self.camera = camera                         # evidence photo source
        self.cnn = ThermalCnn(cfg.get("cnn", {}), data_dir.parent)   # weak-signal model
        self.photo_dir = data_dir / "photos"         # where JPEGs go
        self.photo_dir.mkdir(parents=True, exist_ok=True)   # create it if missing
        self.groups = []                             # open candidate groups
        self.suppressed = []                         # (lat, lon, time) already reported
        self.last_frame = None                       # newest frame, for the UI preview
        self._co2_history = []                       # (time, ppm) for the rolling baseline
        self._co2_last = 0.0                         # last time a sample was appended

    # ---------------------------------------------------------- environment
    def _track_co2(self, now):
        """Append one CO2 sample per second, whatever the thermal frame is doing.

        The baseline must describe the background, so it is sampled continuously —
        not only on frames that happen to contain a cluster. Dropping samples older
        than `co2_baseline_s` keeps the median close to present conditions."""
        co2 = self.sensors.co2.read()                # ppm or None
        if co2 is not None and now - self._co2_last >= 1.0:   # one sample per second
            self._co2_history.append((now, co2))     # add it
            self._co2_last = now                     # remember when
        cutoff = now - self.cfg["co2_baseline_s"]    # oldest sample still worth keeping
        self._co2_history = [s for s in self._co2_history if s[0] >= cutoff]

    def _environment(self, cluster_temp, cnn_prob):
        """Read CO2, reference temperature and moisture; decide if they allow a hit.

        The CO2 gate compares the live reading with the median of the recent baseline,
        so a normally-high background (harbour, car exhaust) does not block everything.
        The temperature gate uses the absolute difference from the DS18B20, which works
        for warm nests at dawn and cool ones at midday. It has two levels: a strong ΔT
        that stands on its own, and a weak ΔT that counts only when the CNN recognises a
        shell pattern. The moisture gate rejects sand that is bone dry or waterlogged,
        because neither can hold a clutch. A sensor that is missing or stale simply
        contributes nothing."""
        co2 = self.sensors.co2.read()                # ppm or None (already sampled above)
        ref_t = self.sensors.temp.read()             # reference °C or None
        moisture = self.sensors.moisture.read()      # percent or None

        baseline = None                              # median background CO2
        if len(self._co2_history) >= 3:              # need a few samples to have a median
            baseline = float(np.median([v for _, v in self._co2_history]))
        rise = None if (co2 is None or baseline is None) else co2 - baseline   # ppm above background
        delta = None if ref_t is None else cluster_temp - ref_t                 # °C above reference
        moisture_ok = None if moisture is None else (self.cfg["moisture_min_pct"]
                                                     <= moisture
                                                     <= self.cfg["moisture_max_pct"])
        # None means "no probe", True/False means "probe says inside/outside the band".

        ok = True                                    # start optimistic
        if self.cfg.get("env_gate", True):           # warm bottle tests switch this off
            if rise is not None and rise < self.cfg["co2_delta_ppm"]:   # not enough CO2
                ok = False                           # conditions do not fit a nest
            if delta is not None:                    # the probe answered
                strong = abs(delta) >= self.cfg["min_delta_c"]       # stands on its own
                weak = (cnn_prob is not None                         # the model answered
                        and cnn_prob >= self.cnn.min_prob            # and sees a shell
                        and abs(delta) >= self.cfg["weak_delta_c"])  # above the noise
                if not (strong or weak):             # neither ΔT path applies
                    ok = False                       # too little contrast, so no nest
            if moisture_ok is False:                 # sand too dry or too wet
                ok = False                           # unlikely to hold a clutch
        return {"ok": ok, "co2": co2, "rise": rise,   # gate result and raw values
                "delta": delta, "moisture": moisture}   # for the detection record

    def _location_score(self):
        """0-1: how well the fix pins this hit to a spot.

        The projected lat/lon is only as good as the NMEA fix behind it, so the score
        combines fix quality (plain GPS < DGPS < RTK), HDOP and satellite count. A
        rover without a GNSS object scores None and the average uses the other inputs."""
        gnss = getattr(self.sensors, "gnss", None)   # the GPS wrapper, if it exists
        if gnss is None:                             # bench tests may omit it
            return None                              # leave this input out
        quality = {1: 0.6, 2: 0.75, 4: 1.0, 5: 0.9}.get(
            getattr(gnss, "quality", 0), 0.4)        # unknown quality is weak evidence
        hdop_ok = _clamp(1.0 - (float(getattr(gnss, "hdop", 99.0)) - 0.8) / 2.2)
        sats_ok = _clamp((int(getattr(gnss, "sats", 0)) - 4) / 8.0)   # 4 -> 0, 12 -> 1
        return (quality + hdop_ok + sats_ok) / 3.0   # all three must agree

    def _scores(self, env, mean_z, consistency, location, cnn_prob):
        """Turn each input into a 0-1 score. Missing sensors are left out, not zeroed.

        A missing sensor must not drag a real nest down, so the confidence is the mean
        of the channels that actually answered. Twice the reference threshold counts as
        a full 1.0. The sand-temperature reference is `weak_delta_c`, because a +0.4 C
        clutch is exactly what this rover is looking for."""
        scores = {"thermal": _clamp(mean_z / (2.0 * self.cfg["z_threshold"]))}
        if env["rise"] is not None:                  # CO2 answered
            need = max(self.cfg["co2_delta_ppm"], 1)   # avoid dividing by zero
            scores["co2"] = _clamp(env["rise"] / (2.0 * need))
        if env["delta"] is not None:                 # reference probe answered
            need = max(self.cfg["weak_delta_c"], 0.05)
            scores["sand_temp"] = _clamp(abs(env["delta"]) / (2.0 * need))
        if env["moisture"] is not None:              # moisture probe answered
            low, high = self.cfg["moisture_min_pct"], self.cfg["moisture_max_pct"]
            margin = min(env["moisture"] - low, high - env["moisture"])
            # A reading just inside the band is weak evidence; 10 % of the band clear
            # of either edge counts as fully plausible.
            scores["moisture"] = _clamp(margin / max(0.1 * (high - low), 1.0))
        scores["consistency"] = _clamp(consistency)   # always available at confirmation
        if location is not None:                     # GNSS answered
            scores["location"] = _clamp(location)
        if cnn_prob is not None:                     # the model answered
            scores["cnn"] = _clamp(cnn_prob)
        return scores                                # one entry per input that spoke

    # ------------------------------------------------------------ geometry
    def _cell_to_ground(self, row, col):
        """(forward_m, right_m) for a thermal pixel, or None if off the ground."""
        vfov = self.mount["vfov_deg"]                # vertical field of view, degrees
        hfov = self.mount["hfov_deg"]                # horizontal field of view, degrees
        rows, cols = self.sensors.thermal.ROWS, self.sensors.thermal.COLS   # 24, 32
        # Row 0 looks nearest the horizon, row 23 looks steeply down. Convert the row
        # index into an angle below the horizon, offset by the mount tilt.
        below = self.mount["tilt_deg"] + (row / (rows - 1) - 0.5) * vfov
        if below <= 6.0:                             # too close to the horizon
            return None                              # no reliable ground intersection
        forward = self.mount["height_m"] / math.tan(math.radians(below))   # flat-ground model
        if not 0.25 <= forward <= 2.0:               # outside the useful scan band
            return None                              # ignore this pixel
        lateral_angle = (col / (cols - 1) - 0.5) * hfov   # column -> left/right angle
        return forward, forward * math.tan(math.radians(lateral_angle))   # metres ahead, metres right

    # -------------------------------------------------------------- cluster
    @staticmethod
    def _largest_cluster(mask):
        """Return the biggest 8-connected blob of True cells, as a list of (row, col)."""
        seen = np.zeros_like(mask, dtype=bool)       # cells already visited
        rows, cols = mask.shape                      # 24, 32
        best = []                                    # cells in the biggest blob so far
        for r in range(rows):                        # scan every row ...
            for c in range(cols):                    # ... every column
                if not mask[r, c] or seen[r, c]:     # empty or already counted
                    continue                         # skip it
                stack, comp = [(r, c)], []           # flood-fill stack and component
                seen[r, c] = True                    # mark the seed
                while stack:                         # depth-first flood fill
                    y, x = stack.pop()               # current cell
                    comp.append((y, x))              # add it to this component
                    for dy in (-1, 0, 1):            # neighbours: ...
                        for dx in (-1, 0, 1):        # ... all eight directions
                            ny, nx = y + dy, x + dx  # neighbour coordinates
                            if (0 <= ny < rows and 0 <= nx < cols       # inside the frame
                                    and mask[ny, nx] and not seen[ny, nx]):  # hot and unvisited
                                seen[ny, nx] = True  # mark it
                                stack.append((ny, nx))   # queue it
                if len(comp) > len(best):            # bigger than the previous best?
                    best = comp                      # keep it
        return best                                  # possibly empty

    def _best_cluster(self, frame, threshold=None):
        """Standardise the frame, then return the strongest warm or cool cluster.

        `threshold` overrides `z_threshold` for the CNN weak-contrast retry."""
        limit = self.cfg["z_threshold"] if threshold is None else threshold
        med = float(np.median(frame))                # middle temperature of the frame
        mad = float(np.median(np.abs(frame - med)))  # median absolute deviation = robust spread
        scale = max(1.4826 * mad, 0.05)              # 1.4826 makes MAD comparable to sigma
        z = (frame - med) / scale                    # every pixel as a robust z-score
        self.last_frame = frame                      # keep the raw frame for the UI preview

        best = None                     # (weight, score, row, col, size, temp, kind)
        for sign, kind in ((1.0, "warm"), (-1.0, "cool")):   # look at both directions
            cells = self._largest_cluster((z * sign) > limit)   # hot side
            if len(cells) < self.cfg["min_cluster_cells"]:   # too small to trust
                continue                                  # try the other direction
            row = sum(c[0] for c in cells) / len(cells)      # centroid row
            col = sum(c[1] for c in cells) / len(cells)      # centroid column
            score = float(np.mean([(z * sign)[r, c] for r, c in cells]))   # mean strength
            temp = float(np.mean([frame[r, c] for r, c in cells]))         # mean °C
            weight = score * len(cells)                      # big + strong beats small + faint
            if best is None or weight > best[0]:             # new winner?
                best = (weight, score, row, col, len(cells), temp, kind)   # keep it

        if best is None:                             # nothing above threshold
            return None                              # no candidate this frame
        _, score, row, col, size, temp, kind = best   # unpack the winner
        return {"row": row, "col": col, "size": size,   # centroid in pixels
                "z": score, "temp": temp, "kind": kind}  # strength, °C, direction

    # --------------------------------------------------------------- output
    def _store_photo(self, now):
        """Save the latest camera frame as evidence. Returns the path or an empty string."""
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + ".jpg"  # timestamp name
        path = self.photo_dir / name                 # full path
        return str(path) if self.camera.snapshot(path) else ""   # path on success, "" otherwise

    def update(self, position, heading):
        """Call at ~8-10 Hz while scanning. Returns a Detection on confirmation."""
        if position is None or heading is None:      # no GPS fix or no compass
            return None                              # cannot geotag anything
        frame = self.sensors.thermal.read()          # one 24x32 temperature frame
        if frame is None:                            # sensor glitch
            return None                              # try again next tick

        now = time.time()                            # wall clock for the log and baseline
        self._track_co2(now)                         # keep the background estimate current

        cnn_prob = self.cnn.predict(frame)           # None if no model or runtime
        cluster = self._best_cluster(frame)          # find the strongest anomaly
        if cluster is None and cnn_prob is not None and cnn_prob >= self.cnn.min_prob:
            # A buried shell at +0.4..0.6 C may not clear the normal threshold. If the
            # model sees the pattern, retry with the faint-signal threshold.
            cluster = self._best_cluster(frame, self.cfg["weak_z_threshold"])
        if cluster is None:                          # nothing unusual in this frame
            return None                              # no candidate

        env = self._environment(cluster["temp"], cnn_prob)   # CO2 / temp / moisture
        if not env["ok"]:                            # conditions do not support a nest
            return None                              # discard this frame's candidate

        ground = self._cell_to_ground(cluster["row"], cluster["col"])   # pixel -> metres
        if ground is None:                           # projected off the sand
            return None                              # ignore it

        lat, lon = ground_to_latlon(position, heading, *ground)   # metres -> GPS coordinate
        radius = self.cfg["cluster_radius_m"]        # how close hits must be to merge

        # Forget suppressions older than the cooldown window.
        self.suppressed = [s for s in self.suppressed
                           if now - s[2] < self.cfg["suppress_time_s"]]
        # Already reported this spot recently? Then do not report it again.
        if any(distance_m((lat, lon), (s[0], s[1])) < self.cfg["suppress_radius_m"]
               for s in self.suppressed):
            return None                              # duplicate, discard

        target = None                                # the group this hit belongs to
        for group in self.groups:                    # check every open candidate
            if distance_m((lat, lon), (group.lat, group.lon)) < radius:   # same spot?
                target = group                       # yes, merge into it
                break                                # stop searching
        if target is None:                           # new spot
            # Seed the running mean with this hit so the first frame is not lost.
            target = _Group(lat, lon, cluster["z"], cluster["temp"],
                            cluster["kind"], now)     # create a group ...
            self.groups.append(target)               # ... and remember it
        else:
            target.add(lat, lon, cluster["z"], cluster["temp"], cluster["kind"], now)  # merge

        # Drop groups that have not been seen for 10 s: the rover has moved on.
        self.groups = [g for g in self.groups if now - g.last < 10.0]

        if target.n < self.cfg["confirm_frames"]:    # not enough agreeing frames yet
            return None                              # wait for more evidence

        # Consistency: the same spot must have given the same temperature and landed in
        # the same place every frame. A wandering warm patch fails here even when each
        # individual frame looked convincing.
        consistency = target.consistency(radius, self.cfg["consistency_temp_c"])
        if consistency < self.cfg["consistency_min"]:   # readings did not agree
            return None                              # keep the group open and re-check

        mean_z = target.z_sum / target.n             # mean strength over the group
        scores = self._scores(env, mean_z, consistency,
                              self._location_score(), cnn_prob)
        confidence = sum(scores.values()) / len(scores)   # mean of the answering inputs

        self.suppressed.append((target.lat, target.lon, now))   # remember not to re-report
        self.groups.remove(target)                   # close the group
        return Detection(
            lat=target.lat, lon=target.lon,          # averaged position
            zscore=mean_z,                           # mean strength
            temp_c=target.temp_sum / target.n,       # mean temperature
            kind=target.kind, frames=target.n,       # direction and frame count
            time=now, photo=self._store_photo(now),  # timestamp and evidence photo
            co2_ppm=env["co2"],                      # CO2 at confirmation (int or None)
            delta_c=env["delta"],                    # cluster minus reference (°C or None)
            moisture_pct=(None if env["moisture"] is None    # whole percent or None
                          else int(env["moisture"])),
            confidence=confidence,                   # 0-1 composite of the inputs
            consistency=consistency,                 # 0-1 repeatability of the frames
            cnn_prob=cnn_prob)                       # CNN shell probability or None

