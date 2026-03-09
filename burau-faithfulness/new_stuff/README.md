# A3 mod p bucket search

This folder contains a type `A_3` version of the finite-ring bucket search from `Bucket_D4`.

Files:

- `setup_a3.py`: oriented Burau data for type `A_3`, together with the dual atoms and the Coxeter element used by the search.
- `a3_mod_p_bucket_search.py`: bucket search over `Z/pZ` using the same spread / dual-Garside framework as the existing `D4` script.

Run from the repository root, for example:

```bash
python3 new_stuff/a3_mod_p_bucket_search.py --p 5 --max-g-length 50 --cpus 4 --stop-at-first
```

Results are written to `new_stuff/data/a3_bucket_hits.json` by default.
