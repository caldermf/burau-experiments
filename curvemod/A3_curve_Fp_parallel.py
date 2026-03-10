from __future__ import annotations

from a3_gpu_search import SearchConfig, run_search


# CAP_1 and CAP_2 are caps on the bucket sizes.
CAP_1 = 500
CAP_2 = 500

# TOTAL_CAP_1 and TOTAL_CAP_2 cap the total number of curves expanded at each step.
TOTAL_CAP_1 = 50000
TOTAL_CAP_2 = 50000

# FIRST_STEPS specifies when to switch from the first caps to the second.
FIRST_STEPS = 12

# PP is the modulus; this script now targets the Fp search only.
PP = 5

# Maximum Garside lengths of braids to consider.
MAX_G_LENGTH = 50

# Device selection: "auto", "cuda", or "cpu".
DEVICE = "auto"

# RNG seed for random-priority bucket capping.
SEED = 0


def main() -> None:
    config = SearchConfig(
        cap_1=CAP_1,
        cap_2=CAP_2,
        total_cap_1=TOTAL_CAP_1,
        total_cap_2=TOTAL_CAP_2,
        first_steps=FIRST_STEPS,
        modulus=PP,
        max_g_length=MAX_G_LENGTH,
        device=DEVICE,
        seed=SEED,
    )
    result = run_search(config)
    if result.found:
        print(f"Found spread-0 witness at depth {result.depth}")
    else:
        print("No spread-0 witness found in the explored range")


if __name__ == "__main__":
    main()
