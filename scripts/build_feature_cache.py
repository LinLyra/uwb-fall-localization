#!/usr/bin/env python3
"""Build deterministic train/validation feature caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from uwb_fall.dataset import extract_split, save_window_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--hop", type=int, default=25)
    parser.add_argument("--range-bands", type=int, default=16)
    args = parser.parse_args()
    for split in ("train", "val"):
        windows = extract_split(
            args.package / "streams",
            args.package / "streams_events.csv",
            split,
            window=args.window,
            hop=args.hop,
            range_bands=args.range_bands,
        )
        target = args.output / f"{split}_w{args.window}_h{args.hop}.npz"
        save_window_set(target, windows)
        positives = int(windows.labels.sum())
        print(
            f"{split}: {len(windows.labels)} windows, {positives} positive "
            f"({positives / len(windows.labels):.2%}), {windows.features.shape[1]} features -> {target}"
        )


if __name__ == "__main__":
    main()
