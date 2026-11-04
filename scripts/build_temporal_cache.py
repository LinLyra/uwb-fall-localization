#!/usr/bin/env python3
"""Build compact order-sensitive CIR window caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from train_tabular_baselines import load_cache
from uwb_fall.temporal import temporal_power_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        metadata = load_cache(args.cache / f"{split}_w100_h25.npz")
        rows: list[np.ndarray] = []
        current = None
        handle = None
        try:
            for stream_id, start in zip(metadata["stream_ids"], metadata["starts"]):
                if stream_id != current:
                    if handle is not None:
                        handle.close()
                    index = str(stream_id).rsplit("_", 1)[-1]
                    handle = h5py.File(args.package / "streams" / split / f"stream_{index}.h5", "r")
                    current = stream_id
                first = int(round(float(start) * 50.0))
                rows.append(temporal_power_sequence(handle["cir"][:, first : first + 100, :]))
        finally:
            if handle is not None:
                handle.close()
        target = args.output / f"{split}_temporal_w100_h25.npz"
        np.savez_compressed(
            target,
            features=np.stack(rows),
            labels=metadata["labels"],
            stream_ids=metadata["stream_ids"],
            starts=metadata["starts"],
            ends=metadata["ends"],
        )
        print(split, len(rows), np.stack(rows).shape, target)


if __name__ == "__main__":
    main()
