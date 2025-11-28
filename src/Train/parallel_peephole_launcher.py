"""Launcher to train multiple node-specific models in parallel.

This script spawns multiple processes that call the existing Hydra-backed
trainer `src.Train.nbe_peephole_lstm_train`. It limits concurrency and
captures stdout/stderr into per-node launcher logs under the specified
`save_dir/{node_id}` directory.

Usage example:
  python src/Train/parallel_peephole_launcher.py \
    --train_dir /path/to/train --val_dir /path/to/val \
    --node_min 1 --node_max 10 --concurrency 4 --save_dir outputs/peephole

You can pass extra Hydra overrides via repeated `--override key=val` flags.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def make_cmd(python: str, train_dir: str, val_dir: str, node_id: int, save_dir: str, overrides: List[str]) -> List[str]:
    # Build module invocation using Hydra overrides (key=value ...)
    cmd = [python, "-m", "src.Train.nbe_peephole_lstm_train"]
    cmd.append(f"train_dir={train_dir}")
    cmd.append(f"val_dir={val_dir}")
    cmd.append(f"node_id={node_id}")
    cmd.append(f"save_dir={save_dir}")
    # append any extra overrides provided by user
    for o in overrides:
        cmd.append(o)
    return cmd


def run_job(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as f:
        # write header
        f.write(("\n--- LAUNCH CMD: %s\n" % (shlex.join(cmd))).encode())
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        ret = proc.wait()
        f.write((f"\n--- EXIT {ret}\n").encode())
    return ret


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--node_min", type=int, required=True)
    p.add_argument("--node_max", type=int, required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--exist_model", action="store_true", help="If set, skip launching for a node when per-node 'best.pth' already exists")
    p.add_argument("--python", default=sys.executable, help="Python executable to run trainer")
    p.add_argument("--save_dir", default="outputs/peephole_train")
    p.add_argument("--override", action="append", default=[], help="Additional Hydra overrides (key=val). Can be repeated.")
    return p.parse_args()


def main():
    args = parse_args()

    node_ids = list(range(args.node_min, args.node_max + 1))
    print(f"Launching training for nodes {args.node_min}..{args.node_max} (total {len(node_ids)}) with concurrency={args.concurrency}")

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {}
        for nid in node_ids:
            per_save = str(Path(args.save_dir) / str(nid))
            # If user requested skipping when a saved model exists, check and continue
            if args.exist_model:
                best_path = Path(per_save)/ str(nid) / "best.pth"
                if best_path.exists():
                    print(f"Skipping node {nid}: model exists at {best_path}")
                    continue
            #cmd = make_cmd(args.python, args.train_dir, args.val_dir, nid, per_save, args.override)
            cmd = make_cmd(args.python, args.train_dir, args.val_dir, nid, args.save_dir, args.override)
            log_path = Path(per_save) / "launcher.log"
            fut = ex.submit(run_job, cmd, log_path)
            futures[fut] = (nid, log_path)

        # collect
        for fut in concurrent.futures.as_completed(futures):
            nid, log_path = futures[fut]
            try:
                rc = fut.result()
            except Exception as e:
                print(f"Node {nid} failed to launch: {e}")
            else:
                print(f"Node {nid} finished with exit code {rc}. launcher log: {log_path}")


if __name__ == "__main__":
    main()
