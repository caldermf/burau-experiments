# Ping-Pong Dataset + MLP

This repo now has a single entrypoint, [`pingpong_mlp.py`](/Users/caldermf/projects/burau-experiments/pingpong/pingpong_mlp.py), for:

- generating raw `R^3` orbit points from alternating nonzero powers of `A` and `B`
- labeling each point by the generator used in the final application step (`A` or `B`)
- training a very small PyTorch MLP on those labels

The label rule is the standard ping-pong rule under the left-action convention used here: after each new matrix power is applied to the current point, the result is labeled by the generator used in that final step. For an actual ping-pong action, those are exactly the two ping-pong domains.

## Quick start

Generate a dataset:

```bash
python3 pingpong_mlp.py generate \
  --v 1.5 \
  --starting-vector 1 1 0 \
  --power-bound 3 \
  --min-length 1 \
  --max-length 20 \
  --num-samples 500000 \
  --output artifacts/pingpong_v1p5.npz
```

Train on a saved dataset:

```bash
python3 pingpong_mlp.py train \
  --dataset artifacts/pingpong_v1p5.npz \
  --epochs 30 \
  --batch-size 4096 \
  --hidden-dims 128 128 \
  --device cuda \
  --checkpoint artifacts/pingpong_mlp_v1p5.pt
```

Train while generating data on the fly:

```bash
python3 pingpong_mlp.py train \
  --v 1.5 \
  --starting-vector 1 1 0 \
  --power-bound 3 \
  --min-length 1 \
  --max-length 20 \
  --train-samples 500000 \
  --val-samples 100000 \
  --epochs 30 \
  --device cuda
```

## Notes

- The generator keeps the literal vectors in `R^3`. It does not project to the sphere.
- Exact enumeration of every alternating word is exponential, so the script uses very fast random sampling of reduced alternating words. That is the practical route for huge datasets.
- The default training feature map is `signed_log1p`, which keeps the raw vectors but compresses extreme coordinate ranges before standardization.
- PyTorch is only needed for the `train` subcommand. Dataset generation uses NumPy only.
