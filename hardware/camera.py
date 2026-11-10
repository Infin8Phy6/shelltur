"""Pi Camera 3 via picamera2.

READING THIS FILE
There is exactly one camera and two consumers: the phone's live video and the evidence
photo saved when a nest candidate is confirmed. Instead of both opening the camera, the
MJPEG encoder writes every frame into a shared "sink" and both consumers read the latest
frame from it. That avoids the classic "camera is busy" error.
"""
import io                          # BufferedIOBase, the interface the encoder expects
import threading                   # Condition, to hand frames between threads safely

try:
    # picamera2 ships with Raspberry Pi OS (apt), not pip.
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder      # compresses frames to JPEG
    from picamera2.outputs import FileOutput         # adapts a file-like object to the encoder
    HAVE_CAMERA = True                               # all three imports worked
except Exception:          # bench testing without a camera
    HAVE_CAMERA = False                              # the rest of the rover still boots


class _Sink(io.BufferedIOBase):
    """A file-like object that keeps only the newest JPEG frame in memory."""

    def __init__(self):
        self.frame = None                            # latest JPEG bytes, or None
        self.condition = threading.Condition()       # lets readers wait for a new frame

    def write(self, buf):
        """Called by the encoder for every encoded frame."""
        with self.condition:                         # lock while touching the shared frame
            self.frame = bytes(buf)                  # copy the framebuffer out
            self.condition.notify_all()              # wake every waiting reader


class Camera:
    """Owns the camera, exposes an MJPEG stream and still snapshots."""

    def __init__(self, cfg):
        self.cfg = cfg                               # width, height, enabled
        self.picam = None                            # set only if the camera starts
        self.sink = _Sink()                          # shared latest-frame store
        if not (HAVE_CAMERA and cfg.get("enabled", True)):
            print("[camera] disabled", flush=True)   # no camera, or turned off in config
            return                                   # the rover still runs without video
        try:
            self.picam = Picamera2()                 # open the camera
            config = self.picam.create_video_configuration(
                main={"size": (cfg["width"], cfg["height"])})  # 1280x720 by default
            self.picam.configure(config)             # apply that resolution
            self.picam.start_recording(MJPEGEncoder(), FileOutput(self.sink))  # stream into the sink
            print("[camera] streaming", flush=True)  # boot log line
        except Exception as exc:                     # ribbon loose, camera in use, etc.
            print(f"[camera] unavailable: {exc}", flush=True)
            self.picam = None                        # pretend there is no camera

    def mjpeg_stream(self):
        """Infinite generator of multipart frames for the Flask /video route."""
        while True:                                  # keep streaming forever
            with self.sink.condition:                # lock the shared frame
                self.sink.condition.wait(timeout=5.0)   # sleep until a new frame (or 5 s)
                frame = self.sink.frame              # grab the newest frame
            if frame:                                # skip if nothing arrived yet
                # Each part is one JPEG inside the multipart boundary the browser expects.
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")

    def snapshot(self, path):
        """Write the latest frame to disk. Returns the path, or None if unavailable."""
        with self.sink.condition:                    # lock the shared frame
            frame = self.sink.frame                  # copy the reference
        if not frame:                                # no frame yet (camera off or still starting)
            return None                              # caller stores an empty photo name
        with open(path, "wb") as handle:             # binary write
            handle.write(frame)                      # save the JPEG
        return path                                  # confirm where it went

    def close(self):
        """Stop recording and release the camera (called on shutdown)."""
        if self.picam:                               # only if we actually opened it
            try:
                self.picam.stop_recording()          # stop the encoder
                self.picam.close()                   # release the device
            except Exception:
                pass                                 # shutdown must never raise
