# Bigelow Burau Search

This repo is self-contained. It does not depend on any code outside `/Users/caldermf/projects/burau-experiments/bigelow`.

## What is implemented

- Exact sparse Laurent polynomials over `ZZ` and `F_p`.
- Exact reduced Burau matrices for braid generators and braid words.
- Exact Artin free-group action helpers for independent nontriviality checks.
- Published witness regressions for:
  - Bigelow's `n=5` kernel element.
  - Bigelow's `n=6` kernel element.
  - The `B_4` `q=2` false-alarm polynomial from the follow-up paper.
- A first-pass orbit search over conjugated twists, plus `n=4` standard-form scaffolding.

## Run the test suite

```bash
python3 -m unittest discover -s tests -v
```

## CLI

```bash
python3 -m bigelow verify-b5
python3 -m bigelow verify-b6
python3 -m bigelow show-b4-q2
python3 -m bigelow orbit-n6
python3 -m bigelow search-n6
```

`search-n6` runs the implemented commuting-pair search at the published depth bounds. It is exact on reported hits because Burau commutation is checked symbolically and nontriviality is checked via the Artin action.
