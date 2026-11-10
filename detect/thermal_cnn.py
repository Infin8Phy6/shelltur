"""Runs the small thermal CNN that looks for buried-shell patterns.

READING THIS FILE
The z-score in nest_detector.py asks one question: "is this patch warmer than the sand
around it?" A shell at 10-20 cm often shows only +0.4 to +0.6 C on the MLX90640, and at
that contrast one threshold cannot separate a clutch from a warm rock, a footprint or a
patch of sunlit sand.

The CNN asks a different question: "does this patch *look* like a buried shell?" It was
trained on 24x32 frames by tools/train_thermal_cnn.py - real frames if you have them,
synthetic weak-signal blobs to start with. Input is the same robust z-score the detector
uses, so a hot afternoon and a cool dawn produce comparable tensors. Output is one
probability in 0..1.

If the model file or the TFLite runtime is missing, predict() returns None and the
detector falls back to its threshold logic. A missing model must never stop a survey.
"""
from pathlib import Path                 # resolve the model path from config

import numpy as np                       # frame maths and tensor shapes


def normalise(frame):
    """Robust z-score of one frame, clipped and scaled to roughly -1..1.

    shape: (1, 24, 32, 1) - batch of one, single channel. The trainer imports this
    exact function, so training and inference cannot drift apart."""
    frame = np.asarray(frame, dtype=np.float32)          # the 24x32 temperature grid
    med = float(np.median(frame))                        # middle temperature
    mad = float(np.median(np.abs(frame - med)))          # robust spread
    scale = max(1.4826 * mad, 0.05)                      # MAD -> sigma, with a floor
    z = np.clip((frame - med) / scale, -5.0, 5.0) / 5.0  # clip outliers, then scale
    return z[None, :, :, None]                           # add batch and channel axes


class ThermalCnn:
    """A tiny TFLite classifier. One instance lives in NestDetector."""

    def __init__(self, cfg, root):
        self.min_prob = float(cfg.get("min_prob", 0.6))   # weak-path threshold
        self.interp = None                                # TFLite interpreter, if loaded
        self._input = None                                # input tensor index
        self._output = None                               # output tensor index
        if not cfg.get("enabled", True):                  # switch in config.yaml
            print("[cnn] disabled", flush=True)
            return
        path = Path(cfg.get("model", "models/thermal_nest.tflite"))
        if not path.is_absolute():                        # relative to the project root
            path = Path(root) / path
        if not path.exists():                             # not trained yet
            print(f"[cnn] no model at {path}; weak-contrast path stays off", flush=True)
            return
        try:
            try:
                from tflite_runtime.interpreter import Interpreter   # small runtime
            except ImportError:                                   # fall back to full TF
                from tensorflow.lite import Interpreter           # big but works
            self.interp = Interpreter(model_path=str(path))
            self.interp.allocate_tensors()                # reserve the tensors once
            self._input = self.interp.get_input_details()[0]["index"]
            self._output = self.interp.get_output_details()[0]["index"]
            print(f"[cnn] {path.name} ready", flush=True)
        except Exception as exc:                          # missing runtime, bad model, etc.
            self.interp = None                            # stay disabled
            print(f"[cnn] unavailable: {exc}", flush=True)

    def predict(self, frame):
        """Probability that the frame contains a buried shell, or None if unavailable."""
        if self.interp is None:                           # no model, no runtime
            return None                                   # caller uses the old gate
        try:
            self.interp.set_tensor(self._input, normalise(frame))
            self.interp.invoke()                          # one forward pass
            return float(self.interp.get_tensor(self._output)[0][0])
        except Exception:                                 # shape mismatch, runtime glitch
            return None                                   # drop this input, never crash a tick
