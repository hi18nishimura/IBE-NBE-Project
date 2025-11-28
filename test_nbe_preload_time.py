#!/usr/bin/env python3
"""Measure NbeDataset preload vs on-demand times and memory.

Usage: python3 test_nbe_preload_time.py --data_dir /path --node_id 1 [--limit 200] [--n_samples 10]
"""
from __future__ import annotations

import argparse
import time
import resource
from pathlib import Path
from statistics import mean

import psutil
import torch

from src.Dataloader.nbeDataset import NbeDataset
from tqdm import tqdm

def rss_mb() -> float:
    # ru_maxrss is in kilobytes on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0


def measure_getitem(ds: NbeDataset, indices: list[int]) -> float:
    times = []
    for i in indices:
        t0 = time.perf_counter()
        item = ds[i]
        t1 = time.perf_counter()
        times.append(t1 - t0)
        # touch tensors to avoid lazy evaluation surprises
        if isinstance(item, dict) and item.get("inputs") is not None:
            _ = item['inputs'].shape
            _ = item['targets'].shape
    return mean(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--node_id", type=int, default=1)
    p.add_argument("--limit", type=int, default=200, help="number of files to preload for partial test (0 = full)")
    p.add_argument("--n_samples", type=int, default=10)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    ds = NbeDataset(data_dir, node_id=args.node_id, preload=False)
    total = len(ds)
    print(f"dataset size: {total} files")

    # baseline: measure on-demand __getitem__ average for n_samples
    indices = list(range(min(args.n_samples, total)))
    print("Measuring on-demand __getitem__ (no preload) ...")
    mem_before = rss_mb()
    t_on_demand = measure_getitem(ds, indices)
    mem_after = rss_mb()
    print(f"on-demand avg sec/sample: {t_on_demand:.4f}, mem {mem_before:.1f} -> {mem_after:.1f} MB")

    # simulate preload: process first `limit` files (or all if limit==0)
    limit = args.limit if args.limit > 0 else total
    limit = min(limit, total)
    print(f"Simulating preload of {limit} files (processing into tensors)...")
    processed = []
    t0 = time.perf_counter()
    mem0 = rss_mb()
    for i, fp in tqdm(enumerate(ds.files[:limit])):
        df = ds._read_file(i)
        node_tables = ds._extract_node_tables(df)
        processed.append(ds._process_node_tables(node_tables, file_path=fp))
    t1 = time.perf_counter()
    mem1 = rss_mb()
    print(f"preload processing time: {t1 - t0:.2f} sec, mem {mem0:.1f} -> {mem1:.1f} MB")

    # attach processed list as cache and measure __getitem__ again
    ds._data_cache = processed
    print("Measuring cached __getitem__ (after preload)...")
    t_cached = measure_getitem(ds, indices)
    mem2 = rss_mb()
    print(f"cached avg sec/sample: {t_cached:.6f}, mem now {mem2:.1f} MB")

    # summary
    print("\nSummary:")
    print(f" on-demand avg sec/sample: {t_on_demand:.4f}")
    print(f" preload processing time ({limit} files): {t1 - t0:.2f} sec")
    print(f" cached avg sec/sample: {t_cached:.6f}")
    print(f" mem before: {mem_before:.1f} MB, after preload: {mem1:.1f} MB, final: {mem2:.1f} MB")


if __name__ == '__main__':
    main()
