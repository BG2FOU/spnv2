import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split sunlamp test.json into train/validation")
    base_default = Path("/home/user-wwy/Dataset/speed-sunlamp")
    parser.add_argument("--input", type=Path, default=base_default / "test.json",
                        help="Path to source JSON (default: /home/user-wwy/Dataset/speed-sunlamp/test.json)")
    parser.add_argument("--train-out", type=Path, default=base_default / "train.json",
                        help="Output path for train split")
    parser.add_argument("--val-out", type=Path, default=base_default / "validation.json",
                        help="Output path for validation split")
    parser.add_argument("--ratio", type=float, default=0.9,
                        help="Train split ratio (0-1)")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for shuffling")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output files if they exist")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")
    if not args.force:
        for out in (args.train_out, args.val_out):
            if out.exists():
                raise FileExistsError(f"Output exists, use --force to overwrite: {out}")

    with args.input.open("r") as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)

    split_idx = int(len(data) * args.ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)

    with args.train_out.open("w") as f:
        json.dump(train_data, f)
    with args.val_out.open("w") as f:
        json.dump(val_data, f)

    print({"train": len(train_data), "validation": len(val_data)})


if __name__ == "__main__":
    main()
