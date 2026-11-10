"""Flask control surface: commands, telemetry, MJPEG and a thermal preview."""
# What this file does: it turns HTTP requests from the phone into messages for the
# control loop, and turns the loop's telemetry dict into JSON. Nothing here touches
# a motor, a serial port or I2C directly — that keeps the web side crash-only.
import io   # in-memory byte buffer, used to build the thermal JPEG without temp files

import numpy as np   # percentile/clip for the false-colour thermal preview
# Flask = routing + templates; Response = streaming bodies; jsonify = dict -> JSON.
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image   # turns a numpy RGB array into a JPEG


def create_app(robot):
    # `robot` is the object built in main.py. Passing it in (instead of importing a
    # global) makes the app testable: hand it a fake robot and nothing else changes.
    # template_folder is relative to this file, so index.html lives in web/templates/.
    app = Flask(__name__, template_folder="templates")

    @app.after_request
    def no_store(response):
        # Hooks run after every view. Live data must never be cached by Safari or
        # Chrome, otherwise the phone shows a frozen speed or an old thermal frame.
        if request.path.startswith(("/api/", "/video", "/thermal")):
            response.headers["Cache-Control"] = "no-store"
        return response             # always return the response or Flask sends nothing

    @app.get("/")
    def index():
        # The single-page phone UI (joystick, buttons, map, camera feed).
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        # The UI polls this about four times a second. snapshot() copies the latest
        # telemetry under a lock, so this call is cheap and thread-safe.
        return jsonify(robot.snapshot())

    @app.post("/api/drive")
    def drive():
        # Joystick endpoint. Format: {"throttle": -1..1, "steer": -1..1}
        data = request.get_json(silent=True) or {}   # missing/bad body -> empty dict
        # Clamp to -1..1 before the value ever reaches the motors: a hostile or buggy
        # client must not be able to request 900% throttle.
        throttle = max(-1.0, min(1.0, float(data.get("throttle", 0.0))))
        steer = max(-1.0, min(1.0, float(data.get("steer", 0.0))))
        # send() only queues the message; it returns immediately even if the loop is busy.
        robot.send({"cmd": "drive", "throttle": throttle, "steer": steer})
        return jsonify(ok=True)     # the UI uses this only to confirm the link is alive

    @app.post("/api/stop")
    def stop():
        # Release the joystick / immediate halt. Also drops the rover out of any goal.
        robot.send({"cmd": "stop"})
        return jsonify(ok=True)

    @app.post("/api/abort")
    def abort():
        # Cancel an autonomous survey and sound two beeps.
        robot.send({"cmd": "abort"})
        return jsonify(ok=True)

    @app.post("/api/arm")
    def arm():
        # "Remember this spot as home." 10 = seconds of GNSS averaging.
        # The loop refuses if there is no fix; the UI shows that as home staying null.
        robot.send({"cmd": "arm_here", "seconds": 10})
        return jsonify(ok=True)

    @app.post("/api/home")
    def home():
        # Drive back to the stored home coordinate (single waypoint mission).
        robot.send({"cmd": "return_home"})
        return jsonify(ok=True)

    @app.post("/api/goto")
    def goto():
        # "Come to my phone": the browser sends its own GPS coordinates here.
        data = request.get_json(silent=True) or {}
        try:
            # float() on a missing key raises KeyError, on null raises TypeError.
            target = {"cmd": "goto", "lat": float(data["lat"]), "lon": float(data["lon"])}
        except (KeyError, TypeError, ValueError):
            # 400 + ok:false is what the UI checks before showing an error toast.
            return jsonify(ok=False, error="lat and lon are required"), 400
        robot.send(target)
        return jsonify(ok=True)

    @app.post("/api/scan")
    def scan():
        # Survey request: two opposite corners of the rectangle, plus optional tuning.
        data = request.get_json(silent=True) or {}
        try:
            payload = {"cmd": "scan",
                       # tl-corner and br-corner; the loop builds the lawnmower grid.
                       "lat1": float(data["lat1"]), "lon1": float(data["lon1"]),
                       "lat2": float(data["lat2"]), "lon2": float(data["lon2"]),
                       # int() keeps a JSON float like 3.0 from becoming a float lane count.
                       "lanes": int(data.get("lanes", 1))}
            if data.get("spacing"):
                # Only override the config's 1.0 m lane spacing if a nonzero value came in.
                payload["spacing"] = float(data["spacing"])
        except (KeyError, TypeError, ValueError):
            # Any missing corner or non-numeric value is a client error, not a server one.
            return jsonify(ok=False, error="lat1/lon1/lat2/lon2 required"), 400
        robot.send(payload)
        return jsonify(ok=True)

    @app.post("/api/estop_reset")
    def estop_reset():
        # Re-arm the motors after the physical e-stop button was released.
        # Refused (ok stays true, but nothing moves) while the button is still latched.
        robot.send({"cmd": "estop_reset"})
        return jsonify(ok=True)

    @app.get("/video")
    def video():
        # Camera feed. mjpeg_stream() is a generator of JPEG frames; Flask streams it
        # as multipart, which every phone browser renders in a plain <img> tag.
        return Response(robot.camera.mjpeg_stream(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/thermal.jpg")
    def thermal():
        # 32x24 false-colour preview of the latest thermal frame, upscaled to 256x192.
        frame = robot.detector.last_frame
        if frame is None:
            # Before the first frame arrives: "not found" makes the <img> show its
            # broken-image state instead of hanging with a half-streamed response.
            return Response(status=404)
        # Stretch the colour scale to this frame: 2nd and 98th percentile ignore the
        # few hottest/coldest cells, so one hot egg cannot wash out the whole picture.
        low, high = np.percentile(frame, 2), np.percentile(frame, 98)
        # Normalise to 0..1. max(..., 0.2) prevents dividing by ~0 on a flat sand frame,
        # and clip keeps any outlier inside the colour ramp.
        norm = np.clip((frame - low) / max(high - low, 0.2), 0.0, 1.0)
        # Three explicit ramps produce blue -> green -> red without a colour-map library.
        rgb = np.dstack([np.clip(1.6 * norm, 0, 1) ** 0.8,          # red rises last
                         np.clip(1.6 * norm - 0.45, 0, 1) ** 1.2,   # green mid
                         np.clip(1.2 - 2.0 * norm, 0, 1)]) * 255    # blue falls first
        # NEAREST keeps the blocky 32x24 look: smoothing would hide single hot pixels.
        image = Image.fromarray(rgb.astype("uint8")).resize((256, 192), Image.NEAREST)
        buffer = io.BytesIO()       # encode in RAM; no file to clean up or lock
        image.save(buffer, "JPEG", quality=80)   # small enough for a slow Wi-Fi link
        return Response(buffer.getvalue(), mimetype="image/jpeg")

    return app      # main.py calls app.run() on this object
