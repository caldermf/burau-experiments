import argparse
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

import setup_a3 as ct


def compute_oburau_vector(braid_word, base_vertex):
    vector = ct.dim_vectors[base_vertex]
    for letter in reversed(braid_word):
        vector = ct.oburau_fns[letter](vector)
    return vector


def compute_oburau_deg(braid_word):
    return max(ct.topdeg_vector(compute_oburau_vector(braid_word, base_vertex)) for base_vertex in ct.positive_letters)


def equal_braids(braid1, braid2):
    return all(
        compute_oburau_vector(braid1, base_vertex) == compute_oburau_vector(braid2, base_vertex)
        for base_vertex in ct.positive_letters
    )


def flatten_factors(factors):
    flat = []
    for factor in factors:
        flat.extend(factor)
    return flat


def make_fp(poly, p):
    if p == 0:
        return poly

    reduced = {}
    for degree, coeff in poly.items():
        new_coeff = coeff % p
        if new_coeff != 0:
            reduced[degree] = new_coeff
    return reduced


def make_fp_vec(vec, p):
    return [make_fp(poly, p) for poly in vec]


def vector_spread(vec):
    return ct.topdeg_vector(vec) - ct.botdeg_vector(vec)


def same_curve_up_to_q_shift(vec, base_vertex):
    return ct.poly_normalize_vector(vec) == ct.dim_vectors[base_vertex]


def find_representative(braid_word, garside_gens):
    for candidate in garside_gens:
        if equal_braids(candidate, braid_word):
            return candidate
    raise ValueError(f"No Garside representative found for {braid_word}")


def build_garside_gens():
    garside_gens = [[]] + ct.DUAL_ATOMS
    additions = 1
    while additions != 0:
        to_add = []
        for left in garside_gens:
            for right in garside_gens:
                candidate = left + right
                if compute_oburau_deg(candidate) > 1:
                    continue

                is_new = True
                for existing in to_add:
                    if equal_braids(existing, candidate):
                        if len(candidate) < len(existing):
                            to_add.remove(existing)
                            to_add.append(candidate)
                        is_new = False
                        break

                if not is_new:
                    continue

                for existing in garside_gens:
                    if equal_braids(existing, candidate):
                        if len(candidate) < len(existing):
                            garside_gens.remove(existing)
                            garside_gens.append(candidate)
                        is_new = False
                        break

                if is_new:
                    to_add.append(candidate)

        garside_gens += to_add
        additions = len(to_add)

    return garside_gens


def build_descents(garside_gens):
    left_descents = {}
    right_descents = {}

    for descent in ct.DUAL_ATOMS:
        for gen in garside_gens:
            if compute_oburau_deg(gen + descent) <= 1:
                rep = find_representative(gen + descent, garside_gens)
                right_descents.setdefault(str(rep), []).append(descent)

            if compute_oburau_deg(descent + gen) <= 1:
                rep = find_representative(descent + gen, garside_gens)
                left_descents.setdefault(str(rep), []).append(descent)

    return left_descents, right_descents


def build_automaton(garside_gens, right_descents, gamma_rep):
    automaton = {}
    for target in garside_gens:
        automaton[str(target)] = []
        if target == [] or target == gamma_rep:
            continue

        for source in garside_gens:
            if source == [] or source == gamma_rep:
                continue

            admissible = True
            for descent in right_descents[str(source)]:
                if descent != [] and descent != gamma_rep and compute_oburau_deg(descent + target) <= 1:
                    admissible = False
                    break

            if admissible:
                automaton[str(target)].append(source)

    return automaton


def initialize_buckets(garside_gens, right_descents, gamma_rep, base_vertex, p):
    buckets = {1: {}}
    for simple in garside_gens:
        if simple == [] or simple == gamma_rep:
            continue

        admissible = True
        for descent in right_descents[str(simple)]:
            if ct.poly_normalize_vector(compute_oburau_vector(descent, base_vertex)) == ct.dim_vectors[base_vertex]:
                admissible = False
                break
            if vector_spread(compute_oburau_vector(descent, base_vertex)) != 1:
                admissible = False
                break

        if not admissible:
            continue

        burau_vec = make_fp_vec(compute_oburau_vector(simple, base_vertex), p)
        spread = vector_spread(burau_vec)
        if spread != 1:
            continue

        buckets[1].setdefault(spread, []).append([[simple], burau_vec])

    return buckets


def iterate_chunk(entrylist, automaton, p, cap, current_length, max_g_length, base_vertex):
    out = {}
    hits = []
    spread_drops = 0

    if entrylist:
        previous_spread = vector_spread(entrylist[0][1])
    else:
        previous_spread = None

    for prev_entry in entrylist:
        leftmost_simple = prev_entry[0][0]
        for next_simple in automaton[str(leftmost_simple)]:
            burau_vec = prev_entry[1].copy()
            for letter in reversed(next_simple):
                burau_vec = make_fp_vec(ct.oburau_fns[letter](burau_vec), p)

            spread = vector_spread(burau_vec)
            if current_length == 2 and spread == 0:
                continue

            normal_form = [next_simple] + prev_entry[0]
            if same_curve_up_to_q_shift(burau_vec, base_vertex):
                hits.append(
                    {
                        "normal_form_factors": normal_form,
                        "artin_word": flatten_factors(normal_form),
                        "spread": spread,
                    }
                )

            if previous_spread is not None and spread < previous_spread:
                spread_drops += 1

            if spread > max_g_length - current_length + 1:
                continue

            if spread in out:
                if len(out[spread]) < cap:
                    out[spread].append([normal_form, burau_vec])
                else:
                    position = random.choice(range(cap + 1))
                    if position < cap:
                        out[spread][position] = [normal_form, burau_vec]
            else:
                out[spread] = [[normal_form, burau_vec]]

    return out, spread_drops, hits


def parse_args():
    parser = argparse.ArgumentParser(description="Bucket search for Burau nonfaithfulness in type A3 modulo p.")
    parser.add_argument("--p", type=int, default=7, help="Work over Z/pZ. Use 0 for Z.")
    parser.add_argument("--max-g-length", type=int, default=1000, help="Maximum Garside length to explore.")
    parser.add_argument("--cpus", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--cap-1", type=int, default=500, help="Per-worker bucket cap during early steps.")
    parser.add_argument("--cap-2", type=int, default=500, help="Per-worker bucket cap after early steps.")
    parser.add_argument("--total-cap-1", type=int, default=50000, help="Total curves retained during early steps.")
    parser.add_argument("--total-cap-2", type=int, default=50000, help="Total curves retained after early steps.")
    parser.add_argument("--first-steps", type=int, default=12, help="Number of early broad-exploration steps.")
    parser.add_argument("--base-vertex", type=int, default=1, choices=ct.positive_letters, help="Which simple root alpha_i to track.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bucket replacement.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "a3_bucket_hits.json", help="Where to write hits and metadata.")
    parser.add_argument("--stop-at-first", action="store_true", help="Stop once a same-curve witness is found.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    start = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    garside_gens = build_garside_gens()
    gamma_rep = find_representative(ct.GAMMA_WORD, garside_gens)
    _, right_descents = build_descents(garside_gens)
    automaton = build_automaton(garside_gens, right_descents, gamma_rep)
    buckets = initialize_buckets(garside_gens, right_descents, gamma_rep, args.base_vertex, args.p)

    hits = []
    current_length = 1
    print(f"Built {len(garside_gens)} Garside simples for A3.")
    print(f"Starting search over Z/{args.p}Z with base vertex {args.base_vertex}.")

    while current_length < args.max_g_length:
        if not buckets[current_length]:
            break

        current_length += 1
        step_start = time.time()
        buckets[current_length] = {}
        spread_drops = 0

        if current_length < args.first_steps:
            total_cap = args.total_cap_1
            cap = args.cap_1
        else:
            total_cap = args.total_cap_2
            cap = args.cap_2

        stop_key = min(buckets[current_length - 1].keys())
        counter = len(buckets[current_length - 1][stop_key])
        while counter < total_cap and stop_key + 1 in buckets[current_length - 1]:
            stop_key += 1
            counter += len(buckets[current_length - 1][stop_key])

        keylist = sorted(key for key in buckets[current_length - 1] if key <= stop_key)
        step_hits = []

        for previous_spread in keylist:
            entries = buckets[current_length - 1][previous_spread]
            print(f"Step {current_length}: analyzing {len(entries)} braids in spread {previous_spread}")
            chunks = [
                [entries[index] for index in range(len(entries)) if index % args.cpus == worker]
                for worker in range(args.cpus)
            ]

            with mp.Pool(args.cpus) as pool:
                results = pool.starmap(
                    iterate_chunk,
                    [
                        (
                            chunk,
                            automaton,
                            args.p,
                            cap,
                            current_length,
                            args.max_g_length,
                            args.base_vertex,
                        )
                        for chunk in chunks
                    ],
                )

            for local_buckets, local_drops, local_hits in results:
                spread_drops += local_drops
                step_hits.extend(local_hits)
                for spread, local_entries in local_buckets.items():
                    if spread not in buckets[current_length]:
                        buckets[current_length][spread] = local_entries.copy()
                        continue

                    if len(buckets[current_length][spread]) < cap * (args.cpus - 1):
                        buckets[current_length][spread] += local_entries
                        continue

                    for entry in local_entries:
                        if len(buckets[current_length][spread]) < cap * args.cpus:
                            buckets[current_length][spread].append(entry)
                        else:
                            position = random.choice(range(args.cpus * cap + 1))
                            if position < args.cpus * cap:
                                buckets[current_length][spread][position] = entry

        hits.extend(step_hits)
        if args.stop_at_first and step_hits:
            break

        buckets[current_length - 1] = {}
        elapsed = time.time() - step_start
        if buckets[current_length]:
            print(
                f"Finished step {current_length}: min spread {min(buckets[current_length].keys())}, "
                f"max spread {max(buckets[current_length].keys())}, drops {spread_drops}, "
                f"hits this step {len(step_hits)}, runtime {elapsed:.2f}s"
            )
        else:
            print(
                f"Finished step {current_length}: no surviving buckets, "
                f"drops {spread_drops}, hits this step {len(step_hits)}, runtime {elapsed:.2f}s"
            )
            break

    payload = {
        "parameters": {
            "p": args.p,
            "max_g_length": args.max_g_length,
            "cpus": args.cpus,
            "cap_1": args.cap_1,
            "cap_2": args.cap_2,
            "total_cap_1": args.total_cap_1,
            "total_cap_2": args.total_cap_2,
            "first_steps": args.first_steps,
            "base_vertex": args.base_vertex,
            "seed": args.seed,
        },
        "num_garside_simples": len(garside_gens),
        "runtime_seconds": time.time() - start,
        "hits": hits,
    }

    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {len(hits)} hits to {args.output}")


if __name__ == "__main__":
    main()
