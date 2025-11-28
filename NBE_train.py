"""Entry point for training. Select model/trainer via --model option.

Usage examples:
  python NBE_train.py --model peepholeLSTM

This will call the `run_training` helper from `src.Train.nbe_peephole_lstm_train`.
"""

import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="peepholeLSTM", help="Which trainer to run")
	args, unknown = parser.parse_known_args()

	if args.model == "peepholeLSTM":
		# import and call the run_training helper
		from src.Train.nbe_peephole_lstm_train import run_training, TrainConfig

		parser.add_argument("--train_dir", type=str, required=True, help="training data directory (feather files)")
		parser.add_argument("--val_dir", type=str, required=True, help="validation data directory (feather files)")
		parser.add_argument("--node_id", type=int, default=1, help="central node id")
		parsed = parser.parse_args()

		cfg = TrainConfig()
		cfg.train_dir = parsed.train_dir
		cfg.val_dir = parsed.val_dir
		cfg.node_id = parsed.node_id
		run_training(cfg)
	else:
		raise ValueError(f"Unknown model/trainer: {args.model}")


if __name__ == "__main__":
	main()
