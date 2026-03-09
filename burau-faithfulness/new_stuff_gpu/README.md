# CUDA-friendly A3 mod p bucket search

This folder is a heavily optimized replacement for `new_stuff`.

What changed:

- live states are stored as dense tensors, not Laurent-dictionary objects
- transitions are applied in batches grouped by leftmost Garside simple
- normal forms are stored with parent pointers instead of carrying full braid words through the search
- the search has a `torch` backend for CUDA devices, with a `numpy` fallback for local development

Files:

- `setup_a3.py`: the same type `A_3` oriented Burau setup as before
- `tensor_backend.py`: backend abstraction for `numpy` or `torch`
- `a3_mod_p_bucket_search.py`: optimized search driver

Recommended cluster usage on an RTX Ada node:

```bash
python a3_mod_p_bucket_search.py \
  --backend torch \
  --device cuda \
  --p 3 \
  --max-g-length 40 \
  --cap-1 500 \
  --cap-2 500 \
  --total-cap-1 50000 \
  --total-cap-2 50000 \
  --first-steps 12 \
  --output data/a3_p3_cuda.json
```

Local CPU fallback:

```bash
python3 new_stuff_gpu/a3_mod_p_bucket_search.py \
  --backend numpy \
  --device cpu \
  --p 3 \
  --max-g-length 20 \
  --stop-at-first
```

Local validation performed here:

- original `new_stuff` first-hit search (`p=3`, `max-g-length=20`) took about `39.5s`
- optimized `new_stuff_gpu` `numpy` backend on the same task took about `3.47s`

The `torch` backend was also validated locally on CPU. On the Yale cluster the intended path is `--backend torch --device cuda`.
