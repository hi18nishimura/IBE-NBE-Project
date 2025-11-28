#!/usr/bin/env python3
"""Measure NbeRelativeDataset data access and DataLoader throughput.

Usage:
  python test_nbe_relative_dataset.py --data_dir /path/to/train --node_id 10

This script measures:
 - avg time for Dataset.__getitem__ over `n_samples`
 - avg time per batch for DataLoader over one epoch
 - reports basic stats and prints JSON summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

try:
    from src.Dataloader.nbeRelativeDataset import NbeRelativeDataset
    from src.Dataloader.nbeRelativeDataset import NbeDataset
except Exception as e:
    print("Could not import NbeRelativeDataset:", e)
    raise


def measure_getitem(ds: NbeRelativeDataset, n_samples: int = 50) -> Dict[str, Any]:
    n = min(len(ds), n_samples)
    times = []
    # warm-up
    for i in range(min(3, n)):
        _ = ds[i]
    # timed
    for i in range(n):
        t0 = time.perf_counter()
        _ = ds[i]
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "requests": n,
        "total_sec": sum(times),
        "avg_sec": mean(times) if times else 0.0,
        "min_sec": min(times) if times else 0.0,
        "max_sec": max(times) if times else 0.0,
    }


def measure_dataloader(ds: NbeRelativeDataset, batch_size: int = 8, num_workers: int = 0, pin_memory: bool = False) -> Dict[str, Any]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    batch_times = []
    n_batches = 0
    # warm-up one pass
    it = iter(loader)
    try:
        next(it)
    except StopIteration:
        return {"n_batches": 0, "total_sec": 0.0, "avg_batch_sec": 0.0}
    # timed epoch
    start = time.perf_counter()
    for batch in loader:
        t0 = time.perf_counter()
        # trivial access to ensure tensors are created
        xb, yb = batch
        # synchronize if cuda
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)
        n_batches += 1
    total = time.perf_counter() - start
    return {
        "n_batches": n_batches,
        "total_sec": total,
        "avg_batch_sec": mean(batch_times) if batch_times else 0.0,
        "min_batch_sec": min(batch_times) if batch_times else 0.0,
        "max_batch_sec": max(batch_times) if batch_times else 0.0,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--node_id", type=int, default=10)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--report", type=str, default=None, help="path to write JSON report")
    args = p.parse_args(argv)

    # choose sensible default data_dir if not provided
    default_dirs = [
        Path("/workspace/tmp_mini_train/train"),
        Path("/workspace/dataset/bin/toy_all_model/train"),
    ]
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None:
        for d in default_dirs:
            if d.exists():
                data_dir = d
                break
    if data_dir is None or not data_dir.exists():
        print("data_dir not found. Provide --data_dir or create /workspace/tmp_mini_train/train")
        sys.exit(2)

    print(f"Using data_dir={data_dir} node_id={args.node_id} preload={args.preload}")

    ds = NbeRelativeDataset(data_dir=str(data_dir), node_id=args.node_id, preload=args.preload)

    print("Dataset size:", len(ds))
    # quick sanity: inspect shapes of first item
    try:
        sample = ds[0]
        print("sample inputs shape:", sample["inputs"].shape, "targets shape:", sample["targets"].shape)
    except Exception as e:
        print("Error reading sample[0]:", e)
        raise

    results: Dict[str, Any] = {}
    print("Measuring __getitem__ ...")
    results["getitem"] = measure_getitem(ds, n_samples=args.n_samples)
    print(json.dumps(results["getitem"], indent=2))

    print("Measuring DataLoader epoch ...")
    results["dataloader"] = measure_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    print(json.dumps(results["dataloader"], indent=2))

    results["config"] = {
        "data_dir": str(data_dir),
        "node_id": args.node_id,
        "preload": args.preload,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }

    if args.report:
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)
        print("Wrote report:", args.report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
