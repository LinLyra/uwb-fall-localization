#!/usr/bin/env python3
"""Build one deterministic robustness augmentation per training window."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from train_tabular_baselines import load_cache
from uwb_fall.features import power_features, shift_range
from uwb_fall.invariance import canonicalize_radar_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    base = load_cache(args.cache / "train_w100_h25.npz")
    rng = np.random.default_rng(args.seed)
    augmented: list[np.ndarray] = []
    kinds: list[str] = []
    current = None
    handle = None
    try:
        for row, (stream_id, start) in enumerate(zip(base["stream_ids"], base["starts"])):
            if stream_id != current:
                if handle is not None:
                    handle.close()
                index = str(stream_id).rsplit("_", 1)[-1]
                handle = h5py.File(args.streams / f"stream_{index}.h5", "r")
                current = stream_id
            assert handle is not None
            first = int(round(float(start) * 50.0))
            values = handle["cir"][:, first : first + 100, :].astype(np.float64)
            bins = values.shape[2] // 2
            i = values[:, :, :bins]
            q = values[:, :, bins:]
            i -= np.mean(i, axis=1, keepdims=True)
            q -= np.mean(q, axis=1, keepdims=True)
            power = (i * i + q * q) * float(rng.uniform(0.7, 1.3))
            if row % 2 == 0:
                offset = int(rng.choice((-16, -8, 8, 16)))
                power = shift_range(power, offset)
                kinds.append(f"shift_{offset:+d}")
            else:
                radar = int(rng.integers(0, 3))
                power[radar] = 0.0
                kinds.append(f"drop_{radar}")
            augmented.append(canonicalize_radar_features(power_features(power)))
    finally:
        if handle is not None:
            handle.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(augmented),
        labels=base["labels"],
        kinds=np.asarray(kinds),
        seed=np.asarray(args.seed),
    )
    print(f"saved {len(augmented)} augmented windows to {args.output}")


if __name__ == "__main__":
    main()
