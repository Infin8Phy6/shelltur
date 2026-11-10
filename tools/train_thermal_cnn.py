"""Trains the tiny thermal CNN used by detect/thermal_cnn.py.

READING THIS FILE
This runs on a laptop, not on the rover. TensorFlow is only needed here; the Pi runs the
exported .tflite file with tflite-runtime, which is a few megabytes.

Two data sources, used together:
  - real frames, saved as .npy from the rover: data/cnn/nest/*.npy and data/cnn/sand/*.npy
    (a file may hold one 24x32 frame or a stack of frames),
  - synthetic frames, generated below. They are not a substitute for your beach, but
    they teach the shape difference immediately: a broad +0.4..0.8 C mound with a faint
    rim (a clutch) versus a small hot rock, a footprint, a broad sun patch or plain noise.

The synthetic generator is deliberately built around the narrow window you measured:
+0.4 to +0.6 C above the surrounding sand. Real frames should be added as soon as you
have them, or the model will learn the simulator instead of the beach.

Usage:
    python tools/train_thermal_cnn.py --synthetic 3000
    python tools/train_thermal_cnn.py --data data/cnn --synthetic 1000 --epochs 60
"""
import argparse                       # command-line options
import sys                            # import the project's normaliser
from pathlib import Path              # paths without string juggling

import numpy as np                    # arrays and the synthetic generator

ROOT = Path(__file__).resolve().parents[1]   # project root (one level above tools/)
sys.path.insert(0, str(ROOT))                 # make `detect` importable
from detect.thermal_cnn import normalise      # training and inference share this


def synthetic_frame(rng, nest):
    """One 24x32 frame in degrees C, labelled nest=True/False.

    The background is a slow ripple plus sensor noise. A nest is a broad, faint bump
    (0.4-0.8 C) with a shallow warmer rim; a shell mound is wide and smooth, unlike the
    sharp point of a rock. Negatives cover the things that actually fool a threshold:
    small hot rocks, elongated footprints, broad sun patches, and featureless noise."""
    yy, xx = np.mgrid[0:24, 0:32]                      # pixel coordinates
    ripple = (0.15 * np.sin(xx / rng.uniform(4, 9) + rng.uniform(0, 6))
              + 0.15 * np.cos(yy / rng.uniform(4, 9) + rng.uniform(0, 6)))
    noise = rng.normal(0, rng.uniform(0.03, 0.12), (24, 32))
    frame = 25.0 + ripple + noise                      # plausible sand temperature
    if nest:
        cy, cx = rng.uniform(6, 18), rng.uniform(8, 24)   # anywhere away from the edge
        sigma = rng.uniform(2.0, 4.5)                  # 4-9 px wide mound
        frame += rng.uniform(0.4, 0.8) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        if rng.random() < 0.7:                         # most nests show a faint rim
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            frame += rng.uniform(0.10, 0.25) * np.exp(
                -((r - 1.8 * sigma) ** 2) / (2 * 1.2 ** 2))
    else:
        kind = int(rng.integers(0, 4))                 # four kinds of false positive
        if kind == 0:                                  # small, hot rock
            cy, cx = rng.integers(3, 21), rng.integers(3, 29)
            frame += rng.uniform(1.5, 4.0) * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / 2.0)
        elif kind == 1:                                # footprint: long and narrow
            cy, cx = rng.integers(6, 18), rng.integers(8, 24)
            frame += rng.uniform(0.3, 0.7) * np.exp(
                -(((yy - cy) ** 2) / (2 * 1.5 ** 2)
                  + ((xx - cx) ** 2) / (2 * 5.0 ** 2)))
        elif kind == 2:                                # broad sun patch
            frame += rng.uniform(0.3, 0.8) * np.exp(
                -(((yy - 12) ** 2 + (xx - 16) ** 2) / (2 * 8.0 ** 2)))
        # kind 3 stays plain noise
    return frame


def load_frames(folder):
    """Every .npy under `folder`, flattened to single frames. Missing folder = empty."""
    frames = []
    for path in sorted(Path(folder).glob("*.npy")):
        arr = np.load(path)                            # (24,32) or (N,24,32)
        if arr.ndim == 2:                              # a single frame
            arr = arr[None]                            # add a batch axis
        frames.extend(arr.reshape(-1, arr.shape[-2], arr.shape[-1]))
    return frames


def augment(tensor, rng):
    """Cheap geometry + noise augmentation on one normalised (24,32,1) tensor."""
    if rng.random() < 0.5:                             # left-right flip
        tensor = tensor[:, ::-1, :]
    if rng.random() < 0.5:                             # up-down flip
        tensor = tensor[::-1, :, :]
    dy, dx = rng.integers(-2, 3, 2)                    # shift by up to 2 px
    tensor = np.roll(np.roll(tensor, dy, 0), dx, 1)
    tensor = tensor + rng.normal(0, 0.03, tensor.shape)   # sensor noise
    return np.clip(tensor, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "cnn"),
                        help="folder with nest/ and sand/ subfolders")
    parser.add_argument("--out", default=str(ROOT / "models" / "thermal_nest.tflite"))
    parser.add_argument("--synthetic", type=int, default=3000,
                        help="synthetic frames to add (0 = none)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        import tensorflow as tf                        # training only
        from tensorflow import keras
    except ImportError:
        sys.exit("TensorFlow is not installed. On the laptop: pip install tensorflow")

    rng = np.random.default_rng(args.seed)             # reproducible data
    data = Path(args.data)

    nest = load_frames(data / "nest")                  # real positive frames
    sand = load_frames(data / "sand")                  # real negative frames
    print(f"real frames: {len(nest)} nest, {len(sand)} sand")
    if args.synthetic:
        nest += [synthetic_frame(rng, True) for _ in range(args.synthetic // 2)]
        sand += [synthetic_frame(rng, False) for _ in range(args.synthetic - args.synthetic // 2)]
        print(f"plus {args.synthetic} synthetic frames")

    if len(nest) == 0 or len(sand) == 0:
        sys.exit("Need at least one frame in both data/cnn/nest and data/cnn/sand")

    x = np.stack([normalise(f) for f in nest + sand]).astype(np.float32)   # (N,24,32,1)
    y = np.array([1.0] * len(nest) + [0.0] * len(sand), dtype=np.float32)

    order = rng.permutation(len(x))                    # shuffle before splitting
    split = int(0.8 * len(x))
    train, test = order[:split], order[split:]

    xtr, ytr = x[train], y[train]
    xte, yte = x[test], y[test]

    extra_x = np.stack([augment(t, rng) for t in xtr for _ in range(2)])   # 3x data
    xtr = np.concatenate([xtr, extra_x])
    ytr = np.concatenate([ytr, np.repeat(ytr, 2)])     # match the pair-per-frame order

    model = keras.Sequential([
        keras.layers.Input((24, 32, 1)),
        keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.GlobalAveragePooling2D(),         # keeps the model tiny
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),   # probability of "nest"
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(xtr, ytr, validation_split=0.15, epochs=args.epochs, batch_size=32,
              callbacks=[keras.callbacks.EarlyStopping(patience=8,
                                                       restore_best_weights=True)],
              verbose=2)

    prob = model.predict(xte, verbose=0).ravel()       # held-out probabilities
    for gate in (0.5, 0.6, 0.7):                       # show the trade-off at the gate
        pred = prob >= gate
        acc = float(np.mean(pred == (yte == 1)))
        print(f"threshold {gate:.1f}: accuracy {acc:.3f}")

    out = Path(args.out)                               # export location
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out.with_suffix(".keras")))         # keep the trainable model too

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # small enough for the Pi
    out.write_bytes(converter.convert())
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
