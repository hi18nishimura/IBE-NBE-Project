
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def find_best_checkpoint(root: Path, model_name: str = "best.pth") -> Optional[Path]:
	"""Recursively search for a file named `model_name` (default 'best.pth') under `root`.

	Returns the first match as a Path, or None if not found.
	"""
	root = Path(root)
	if not root.exists():
		return None
	# use rglob to search recursively
	for p in root.rglob(model_name):
		if p.is_file():
			return p
	return None


def load_best_checkpoint(
	root: str | Path,
	map_location: Optional[torch.device | str] = None,
	weights_only: bool = True,
	model_name: str = "best.pth",
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
	"""Load checkpoint dict from the first `model_name` under `root`.

	This function attempts a weights-only load by default (safe, tensors only).
	If that fails and `weights_only` is True we will retry a full load with
	a permissive safe-globals context for common non-tensor objects (e.g. OmegaConf)
	when available. Returns (checkpoint_dict or None, path_to_ckpt or None).
	"""
	ckpt_path = find_best_checkpoint(Path(root), model_name=model_name)
	if ckpt_path is None:
		return None, None
	if map_location is None:
		map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Try weights-only load first (supported in PyTorch >=2.6). If the current
	# torch implementation does not accept weights_only, fall back to the
	# older signature without that argument.
	try:
		try:
			ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=weights_only)
			return ckpt, ckpt_path
		except TypeError:
			# torch.load doesn't accept weights_only (older torch). Call without it.
			ckpt = torch.load(str(ckpt_path), map_location=map_location)
			return ckpt, ckpt_path
	except Exception as exc:
		# If weights_only mode failed and the caller allowed a full load, try
		# again with weights_only=False and, if available, allow OmegaConf types.
		if not weights_only:
			raise
		try:
			import omegaconf

			safe_ctx = getattr(torch.serialization, "safe_globals", None)
			add_safe = getattr(torch.serialization, "add_safe_globals", None)
			if safe_ctx is not None:
				with safe_ctx([omegaconf.base.ContainerMetadata]):
					ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
			elif add_safe is not None:
				# modify global allowlist then load
				add_safe([omegaconf.base.ContainerMetadata])
				ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
			else:
				# last resort: attempt full load (may raise UnpicklingError)
				ckpt = torch.load(str(ckpt_path), map_location=map_location)
			return ckpt, ckpt_path
		except Exception:
			# Re-raise the original exception for diagnostics
			raise exc


def build_model_from_ckpt(
	ckpt: Dict[str, Any],
	model_ctor,  # callable to construct the model, e.g. NbePeepholeLSTM
	model_ctor_kwargs: Optional[Dict[str, Any]] = None,
	map_location: Optional[torch.device | str] = None,
) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]]]:
	"""Construct a model using `model_ctor(**model_ctor_kwargs)` and load weights from `ckpt`.

	Handles common checkpoint layouts where the state dict is stored under
	keys like 'model_state' or 'state_dict'. Returns (model_or_None, loaded_ckpt_dict_or_None).
	"""
	if ckpt is None:
		return None, None
	model_ctor_kwargs = model_ctor_kwargs or {}
	model = model_ctor(**model_ctor_kwargs)
	# decide device for loading
	if map_location is None:
		map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# find likely state dict key
	state = None
	for k in ("model_state", "state_dict", "model_state_dict"):
		if k in ckpt:
			state = ckpt[k]
			break
	if state is None:
		# maybe the checkpoint itself *is* a state_dict
		if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
			state = ckpt
		else:
			# unknown layout
			# try to be permissive: if a top-level key 'model_state' missing, try to look
			# for any dict-like value that looks like a state_dict
			for v in ckpt.values():
				if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
					state = v
					break
	if state is None:
		# nothing we can load
		return model, ckpt
	try:
		model.load_state_dict(state, strict=False)
	except Exception as e:
		# attempt to convert keys if saved with 'module.' prefixes
		new_state = {k.replace("module.", ""): v for k, v in state.items()}
		model.load_state_dict(new_state, strict=False)
	model.to(map_location)
	model.eval()
	return model, ckpt


def load_model_from_dir(
	root: str | Path,
	model_ctor,
	model_ctor_kwargs: Optional[Dict[str, Any]] = None,
	map_location: Optional[torch.device | str] = None,
	model_name: str = "best.pth",
) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]], Optional[Path]]:
	"""Convenience: search for `model_name`, load checkpoint, build model and load weights.

	Returns (model or None, ckpt dict or None, ckpt_path or None).
	"""
	ckpt, path = load_best_checkpoint(root, map_location=map_location, weights_only=False, model_name=model_name)
	if ckpt is None:
		return None, None, None
	model, ckpt_loaded = build_model_from_ckpt(ckpt, model_ctor, model_ctor_kwargs, map_location=map_location)
	return model, ckpt_loaded, path


if __name__ == "__main__":
	# simple CLI for manual testing
	import argparse

	from src.Networks.nbe_peephole_lstm import NbePeepholeLSTM

	parser = argparse.ArgumentParser(description="Load best.pth recursively and attach to NbePeepholeLSTM")
	parser.add_argument("root_dir", help="Directory to search for best.pth")
	parser.add_argument("--input_size", type=int, required=True)
	parser.add_argument("--hidden_size", type=int, required=True)
	parser.add_argument("--output_size", type=int, default=None)
	parser.add_argument("--num_layers", type=int, default=1)
	parser.add_argument("--dropout", type=float, default=0.0)
	parser.add_argument("--device", default=None, help="torch device string, e.g. cpu or cuda:0")
	args = parser.parse_args()

	device = torch.device(args.device) if args.device else None

	model_ctor_kwargs = {
		"input_size": args.input_size,
		"hidden_size": args.hidden_size,
		"output_size": args.output_size,
		"num_layers": args.num_layers,
		"dropout": args.dropout,
	}

	model, ckpt, path = load_model_from_dir(args.root_dir, NbePeepholeLSTM, model_ctor_kwargs, map_location=device)
	if model is None:
		print("No checkpoint found under", args.root_dir)
		raise SystemExit(2)
	print("Loaded checkpoint:", path)
	print("Model summary:")
	print(model)
