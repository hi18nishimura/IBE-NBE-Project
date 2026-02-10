"""Launcher to train multiple node-specific AutoEncoder MLP models in parallel.

This script spawns multiple processes that call the existing Hydra-backed
trainer `src.Train.nbe_autoencoder_mlp_train`. It limits concurrency and
captures stdout/stderr into per-node launcher logs under the specified
`save_dir/{node_id}` directory.

Usage example:
  python src/Train/parallel_autoencoder_mlp_launcher.py \
    --train_dir /path/to/train --val_dir /path/to/val \
    --node_min 1 --node_max 10 --concurrency 4 --save_dir outputs/autoencoder_mlp_train \
    --config-name autoencoder_mlp

You can pass extra Hydra overrides via repeated `--override key=val` flags.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shlex
import subprocess
import sys
import torch
from pathlib import Path
from typing import List, Optional
from omegaconf import OmegaConf


def make_cmd(python: str, train_dir: Optional[str], val_dir: Optional[str], node_id: int, save_dir: Optional[str], config_name: str, overrides: List[str]) -> List[str]:
    # Build module invocation using Hydra overrides (key=value ...)
    cmd = [python, "-m", "src.Train.nbe_autoencoder_mlp_train"]
    
    # Add config name if provided
    if config_name:
        cmd.append(f"--config-name={config_name}")
        
    if train_dir:
        cmd.append(f"train_dir={train_dir}")
    if val_dir:
        cmd.append(f"val_dir={val_dir}")
    
    cmd.append(f"node_id={node_id}")
    
    if save_dir:
        cmd.append(f"save_dir={save_dir}")

    # append any extra overrides provided by user
    for o in overrides:
        cmd.append(o)
    return cmd


def run_job(cmd: List[str], log_path: Path, gpu_id: Optional[int] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set specific GPU if requested
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with open(log_path, "ab", buffering=0) as f:
        # write header
        f.write(("\n--- LAUNCH CMD: %s\n" % (shlex.join(cmd))).encode())
        if gpu_id is not None:
            f.write(("\n--- GPU ID: %d\n" % gpu_id).encode())
            
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        ret = proc.wait()
        f.write((f"\n--- EXIT {ret}\n").encode())
    return ret


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=False, default=None, help="Path to training data. If not provided, uses value from config.")
    p.add_argument("--val_dir", required=False, default=None, help="Path to validation data. If not provided, uses value from config.")
    p.add_argument("--node_min", type=int, required=True)
    p.add_argument("--node_max", type=int, required=True)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--exist_model", action="store_true", help="If set, skip launching for a node when per-node 'best.pth' already exists")
    p.add_argument("--python", default=sys.executable, help="Python executable to run trainer")
    p.add_argument("--save_dir", required=False, default=None, help="Output directory. If not provided, uses value from config.")
    p.add_argument("--config-name", default="autoencoder_mlp", help="Hydra config name (e.g. autoencoder_mlp)")
    p.add_argument("--override", action="append", default=[], help="Additional Hydra overrides (key=val). Can be repeated.")
    return p.parse_args()


def main():
    args = parse_args()

    # Determine save_dir for launcher logging
    # If not provided via CLI, read from YAML config to ensure logs go to the right place
    launcher_save_dir = args.save_dir
    if launcher_save_dir is None:
        try:
            # Assume config is in config/NBE/{config_name}.yaml relative to CWD
            config_path = Path("config/NBE") / f"{args.config_name}.yaml"
            if config_path.exists():
                cfg = OmegaConf.load(config_path)
                # Handle possible @package _global_ structure or direct keys
                launcher_save_dir = cfg.get("save_dir", "outputs/autoencoder_mlp_train")
                print(f"Loaded save_dir from config: {launcher_save_dir}")
            else:
                print(f"Warning: Config file not found at {config_path}. Using default output path.")
                launcher_save_dir = "outputs/autoencoder_mlp_train"
        except Exception as e:
            print(f"Error reading config: {e}. Using default output path.")
            launcher_save_dir = "outputs/autoencoder_mlp_train"

    node_ids = list(range(args.node_min, args.node_max + 1))
    print(f"Launching AutoEncoder MLP training for nodes {args.node_min}..{args.node_max} (total {len(node_ids)}) with concurrency={args.concurrency}")

    device_cnt = torch.cuda.device_count()
    print(f"Available GPUs: {device_cnt}")

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {}
        for i, nid in enumerate(node_ids):
            per_save = str(Path(launcher_save_dir) / str(nid))
            # If user requested skipping when a saved model exists, check and continue
            if args.exist_model:
                best_path = Path(per_save) / "best.pth"
                if best_path.exists():
                    print(f"Skipping node {nid}: model exists at {best_path}")
                    continue
            
            # Pass args.save_dir (which might be None) to let child process determine output if not overridden
            cmd = make_cmd(args.python, args.train_dir, args.val_dir, nid, args.save_dir, args.config_name, args.override)
            log_path = Path(per_save) / "launcher.log"
            
            # Assign GPU round-robin based on index
            gpu_id = i % device_cnt if device_cnt > 0 else None
            
            fut = ex.submit(run_job, cmd, log_path, gpu_id)
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
