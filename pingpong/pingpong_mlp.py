#!/usr/bin/env python3
"""
Minimal ping-pong dataset generator and MLP trainer.

The label convention follows the ping-pong lemma under the same left-action
convention used by the code: at each step we apply one more nonzero power of A
or B to the current point, and we label the result by the generator used in the
final application step. For a genuine ping-pong action, that is the side of the
ping-pong domain containing the point.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


LABEL_NAMES = np.array(["A", "B"])


@dataclass
class GeneratorConfig:
    v: float = 1.5
    starting_vector: tuple[float, float, float] = (1.0, 1.0, 0.0)
    power_bound: int = 3
    min_length: int = 1
    max_length: int = 16
    num_samples: int = 200_000
    chunk_size: int = 50_000
    seed: int = 0
    dtype: str = "float64"
    renormalize_b: bool = False


def resolve_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float64":
        return np.dtype(np.float64)
    if dtype_name == "longdouble":
        return np.dtype(np.longdouble)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def validate_generator_config(config: GeneratorConfig) -> None:
    if config.power_bound < 1:
        raise ValueError("power_bound must be at least 1")
    if config.min_length < 1:
        raise ValueError("min_length must be at least 1")
    if config.max_length < config.min_length:
        raise ValueError("max_length must be at least min_length")
    if config.num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if config.chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if np.linalg.norm(np.asarray(config.starting_vector, dtype=float)) == 0.0:
        raise ValueError("starting_vector must be nonzero")


def build_generators(
    v: float,
    *,
    renormalize_b: bool = False,
    dtype_name: str = "float64",
) -> tuple[np.ndarray, np.ndarray]:
    dtype = resolve_dtype(dtype_name).type
    v_typed = dtype(v)

    A = np.array(
        [[v_typed**2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, v_typed ** (-2)]],
        dtype=dtype,
    )
    q3 = (v_typed**3 - v_typed ** (-3)) / (v_typed - v_typed ** (-1))
    q2 = (v_typed**2 - v_typed ** (-2)) / (v_typed - v_typed ** (-1))
    q1 = dtype(1.0)
    B = np.array([[0.0, q3, q2], [0.0, q2, q1], [-1.0, 0.0, 0.0]], dtype=dtype)

    if renormalize_b:
        B = B.copy()
        B[0, 1] *= v_typed ** (-1)
        B[1, 2] *= v_typed

    return A, B


def nonzero_exponents(power_bound: int) -> np.ndarray:
    negative = np.arange(-power_bound, 0, dtype=np.int64)
    positive = np.arange(1, power_bound + 1, dtype=np.int64)
    return np.concatenate([negative, positive])


def precompute_powers(
    A: np.ndarray,
    B: np.ndarray,
    power_bound: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exponents = nonzero_exponents(power_bound)
    A_powers = np.stack([np.linalg.matrix_power(A, int(exp)) for exp in exponents], axis=0)
    B_powers = np.stack([np.linalg.matrix_power(B, int(exp)) for exp in exponents], axis=0)
    return exponents, A_powers, B_powers


def batch_apply(matrices: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", matrices, vectors, optimize=True)


def apply_alternating_words(
    starting_vector: np.ndarray,
    start_generators: np.ndarray,
    lengths: np.ndarray,
    power_choice_indices: np.ndarray,
    A_powers: np.ndarray,
    B_powers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if power_choice_indices.shape[0] != lengths.shape[0]:
        raise ValueError("power_choice_indices and lengths must have the same batch size")
    if power_choice_indices.shape[0] != start_generators.shape[0]:
        raise ValueError("power_choice_indices and start_generators must have the same batch size")

    batch_size = lengths.shape[0]
    max_length = power_choice_indices.shape[1]
    points = np.repeat(starting_vector.reshape(1, -1), batch_size, axis=0)

    for step in range(max_length):
        active_indices = np.flatnonzero(lengths > step)
        if active_indices.size == 0:
            break

        active_generators = (start_generators[active_indices] + step) & 1

        a_indices = active_indices[active_generators == 0]
        if a_indices.size:
            a_choice_indices = power_choice_indices[a_indices, step]
            points[a_indices] = batch_apply(A_powers[a_choice_indices], points[a_indices])

        b_indices = active_indices[active_generators == 1]
        if b_indices.size:
            b_choice_indices = power_choice_indices[b_indices, step]
            points[b_indices] = batch_apply(B_powers[b_choice_indices], points[b_indices])

    labels = ((start_generators + lengths - 1) & 1).astype(np.int64)
    return points, labels


def generate_dataset(config: GeneratorConfig) -> dict[str, Any]:
    validate_generator_config(config)

    A, B = build_generators(
        config.v,
        renormalize_b=config.renormalize_b,
        dtype_name=config.dtype,
    )
    exponents, A_powers, B_powers = precompute_powers(A, B, config.power_bound)
    dtype = A.dtype

    rng = np.random.default_rng(config.seed)

    points = np.empty((config.num_samples, 3), dtype=dtype)
    labels = np.empty(config.num_samples, dtype=np.int64)
    lengths = np.empty(config.num_samples, dtype=np.int64)
    start_generators = np.empty(config.num_samples, dtype=np.int64)

    starting_vector = np.asarray(config.starting_vector, dtype=dtype)
    write_index = 0
    discarded = 0

    with np.errstate(over="ignore", invalid="ignore"):
        while write_index < config.num_samples:
            batch_size = min(config.chunk_size, config.num_samples - write_index)
            sampled_lengths = rng.integers(
                config.min_length,
                config.max_length + 1,
                size=batch_size,
                dtype=np.int64,
            )
            sampled_starts = rng.integers(0, 2, size=batch_size, dtype=np.int64)
            sampled_power_choices = rng.integers(
                0,
                len(exponents),
                size=(batch_size, config.max_length),
                dtype=np.int64,
            )

            batch_points, batch_labels = apply_alternating_words(
                starting_vector,
                sampled_starts,
                sampled_lengths,
                sampled_power_choices,
                A_powers,
                B_powers,
            )
            valid_mask = np.isfinite(batch_points).all(axis=1)
            if not np.any(valid_mask):
                raise RuntimeError(
                    "All generated samples overflowed or became invalid. "
                    "Lower max_length or power_bound, or reduce v."
                )

            valid_indices = np.flatnonzero(valid_mask)
            discarded += batch_size - valid_indices.size
            take = min(valid_indices.size, config.num_samples - write_index)
            valid_indices = valid_indices[:take]

            next_write = write_index + take
            points[write_index:next_write] = batch_points[valid_indices]
            labels[write_index:next_write] = batch_labels[valid_indices]
            lengths[write_index:next_write] = sampled_lengths[valid_indices]
            start_generators[write_index:next_write] = sampled_starts[valid_indices]
            write_index = next_write

    return {
        "points": points,
        "labels": labels,
        "lengths": lengths,
        "start_generators": start_generators,
        "label_names": LABEL_NAMES,
        "exponents": exponents,
        "A": A,
        "B": B,
        "discarded_invalid": np.int64(discarded),
        "config_json": json.dumps(asdict(config)),
    }


def save_dataset(path: Path, dataset: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def load_dataset(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        loaded: dict[str, Any] = {key: data[key] for key in data.files}
    if "config_json" in loaded and isinstance(loaded["config_json"], np.ndarray):
        loaded["config_json"] = loaded["config_json"].item()
    return loaded


def format_summary(points: np.ndarray, labels: np.ndarray) -> str:
    norms = np.linalg.norm(points, axis=1)
    count_a = int(np.sum(labels == 0))
    count_b = int(np.sum(labels == 1))
    return (
        f"samples={len(points)} "
        f"A={count_a} "
        f"B={count_b} "
        f"norm_range=({norms.min():.4g}, {norms.max():.4g})"
    )


def feature_map(points: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return np.asarray(points, dtype=np.float64)
    if mode == "signed_log1p":
        points64 = np.asarray(points, dtype=np.float64)
        return np.sign(points64) * np.log1p(np.abs(points64))
    raise ValueError(f"Unsupported feature map: {mode}")


def add_generator_args(parser: argparse.ArgumentParser, *, include_num_samples: bool) -> None:
    parser.add_argument("--v", type=float, default=1.5, help="Specialization parameter.")
    parser.add_argument(
        "--starting-vector",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(1.0, 1.0, 0.0),
        help="Starting vector in R^3.",
    )
    parser.add_argument("--power-bound", type=int, default=3, help="Use exponents in [-p, -1] U [1, p].")
    parser.add_argument("--min-length", type=int, default=1, help="Minimum syllable length.")
    parser.add_argument("--max-length", type=int, default=16, help="Maximum syllable length.")
    if include_num_samples:
        parser.add_argument("--num-samples", type=int, default=200_000, help="Number of samples to generate.")
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Generation chunk size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--dtype",
        choices=("float64", "longdouble"),
        default="float64",
        help="NumPy dtype used during generation.",
    )
    parser.add_argument(
        "--renormalize-b",
        action="store_true",
        help="Match the original notebook's ad-hoc B renormalization.",
    )


def config_from_args(args: argparse.Namespace, *, num_samples: int) -> GeneratorConfig:
    return GeneratorConfig(
        v=args.v,
        starting_vector=tuple(args.starting_vector),
        power_bound=args.power_bound,
        min_length=args.min_length,
        max_length=args.max_length,
        num_samples=num_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
        dtype=args.dtype,
        renormalize_b=args.renormalize_b,
    )


def pick_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for `train`. Install a CUDA-capable torch build in "
            "your GPU environment, then re-run the training command."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def train_model(args: argparse.Namespace) -> None:
    torch, nn, DataLoader, TensorDataset = require_torch()

    if args.dataset is not None:
        loaded = load_dataset(Path(args.dataset))
        all_points = loaded["points"]
        all_labels = loaded["labels"]
        if len(all_points) < 2:
            raise ValueError("Dataset must contain at least two samples")
        rng = np.random.default_rng(args.seed)
        permutation = rng.permutation(len(all_points))
        val_size = max(1, int(round(len(all_points) * args.val_fraction)))
        if val_size >= len(all_points):
            val_size = len(all_points) - 1
        val_indices = permutation[:val_size]
        train_indices = permutation[val_size:]
        train_points = all_points[train_indices]
        train_labels = all_labels[train_indices]
        val_points = all_points[val_indices]
        val_labels = all_labels[val_indices]
        dataset_config = loaded.get("config_json")
    else:
        train_config = config_from_args(args, num_samples=args.train_samples)
        val_config = config_from_args(args, num_samples=args.val_samples)
        val_config.seed = args.seed + 1
        train_data = generate_dataset(train_config)
        val_data = generate_dataset(val_config)
        train_points = train_data["points"]
        train_labels = train_data["labels"]
        val_points = val_data["points"]
        val_labels = val_data["labels"]
        dataset_config = {"train": asdict(train_config), "val": asdict(val_config)}

    train_features = feature_map(train_points, args.feature_map)
    val_features = feature_map(val_points, args.feature_map)

    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True)
    feature_std[feature_std == 0.0] = 1.0

    train_features = ((train_features - feature_mean) / feature_std).astype(np.float32)
    val_features = ((val_features - feature_mean) / feature_std).astype(np.float32)
    train_targets = train_labels.astype(np.float32)
    val_targets = val_labels.astype(np.float32)

    device = pick_device(torch, args.device)

    train_dataset = TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_targets))
    val_dataset = TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_targets))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    hidden_dims = tuple(args.hidden_dims)
    layers: list[Any] = []
    input_dim = train_features.shape[1]
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(nn.ReLU())
        if args.dropout > 0.0:
            layers.append(nn.Dropout(args.dropout))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, 1))
    model = nn.Sequential(*layers).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    def evaluate(loader: Any) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                logits = model(batch_features).squeeze(1)
                loss = criterion(logits, batch_targets)
                predictions = (logits >= 0.0).float()
                total_loss += float(loss.item()) * batch_targets.shape[0]
                total_correct += int((predictions == batch_targets).sum().item())
                total_examples += batch_targets.shape[0]
        return total_loss / total_examples, total_correct / total_examples

    best_val_accuracy = -1.0
    best_checkpoint: dict[str, Any] | None = None

    print(f"train {format_summary(train_points, train_labels)}")
    print(f"val   {format_summary(val_points, val_labels)}")
    print(f"device={device} feature_map={args.feature_map} hidden_dims={hidden_dims}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).squeeze(1)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * batch_targets.shape[0]
            running_examples += batch_targets.shape[0]

        train_loss = running_loss / running_examples
        val_loss, val_accuracy = evaluate(val_loader)
        _, train_accuracy = evaluate(train_loader)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.5f} "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "hidden_dims": list(hidden_dims),
                "dropout": args.dropout,
                "feature_map": args.feature_map,
                "feature_mean": feature_mean.astype(np.float32),
                "feature_std": feature_std.astype(np.float32),
                "dataset_config": dataset_config,
                "training_args": vars(args),
                "label_names": LABEL_NAMES.tolist(),
                "best_val_accuracy": best_val_accuracy,
            }

    if args.checkpoint is not None and best_checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_checkpoint, checkpoint_path)
        print(f"saved checkpoint to {checkpoint_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate a labeled ping-pong dataset.")
    add_generator_args(generate_parser, include_num_samples=True)
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pingpong_dataset.npz"),
        help="Output .npz path.",
    )

    train_parser = subparsers.add_parser("train", help="Train a simple MLP classifier.")
    add_generator_args(train_parser, include_num_samples=False)
    train_parser.add_argument("--dataset", type=Path, default=None, help="Existing .npz dataset to train on.")
    train_parser.add_argument(
        "--train-samples",
        type=int,
        default=400_000,
        help="Training sample count when generating on the fly.",
    )
    train_parser.add_argument(
        "--val-samples",
        type=int,
        default=100_000,
        help="Validation sample count when generating on the fly.",
    )
    train_parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction when --dataset is provided.",
    )
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=4096, help="Batch size.")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    train_parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    train_parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer widths.",
    )
    train_parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")
    train_parser.add_argument(
        "--feature-map",
        choices=("raw", "signed_log1p"),
        default="signed_log1p",
        help="Input preprocessing before standardization.",
    )
    train_parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, or mps.",
    )
    train_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/pingpong_mlp.pt"),
        help="Where to save the best checkpoint.",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "generate":
        config = config_from_args(args, num_samples=args.num_samples)
        dataset = generate_dataset(config)
        save_dataset(args.output, dataset)
        print(f"saved dataset to {args.output}")
        print(format_summary(dataset["points"], dataset["labels"]))
        print(f"discarded_invalid={int(dataset['discarded_invalid'])}")
        return

    if args.command == "train":
        train_model(args)
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
