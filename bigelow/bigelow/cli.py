from __future__ import annotations

import argparse

from .algebra import FiniteFieldPrime, ZZ
from .braid import burau_word_matrix, first_noncommuting_generator_image
from .search import SearchConfig, WitnessType, find_commuting_kernel_pair, orbit_search_states
from .witnesses import (
    bigelow_b4_q2_polynomial,
    bigelow_b5_left_twist_word,
    bigelow_b5_right_twist_word,
    bigelow_b6_left_conjugator,
    bigelow_b6_left_twist_word,
    bigelow_b6_right_conjugator,
    bigelow_b6_right_twist_word,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Correctness-first Burau experiments in the self-contained bigelow repo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify-b5", help="Verify Bigelow's published n=5 kernel witness over ZZ[q,q^-1].")
    subparsers.add_parser("verify-b6", help="Verify Bigelow's published n=6 kernel witness over ZZ[q,q^-1].")
    subparsers.add_parser("show-b4-q2", help="Show the published B4 q=2 false-alarm polynomial.")
    subparsers.add_parser("orbit-n6", help="Enumerate the n=6 left-hand twist orbit up to the published depth.")

    search_n6 = subparsers.add_parser("search-n6", help="Attempt a full n=6 commuting-pair search at the published depths.")
    search_n6.add_argument("--mod-p", type=int, default=None, help="Optional coefficient reduction prime.")

    return parser


def cmd_verify_b5() -> int:
    left = burau_word_matrix(5, bigelow_b5_left_twist_word(), ZZ)
    right = burau_word_matrix(5, bigelow_b5_right_twist_word(), ZZ)
    witness = first_noncommuting_generator_image(5, bigelow_b5_left_twist_word(), bigelow_b5_right_twist_word())
    print(f"Burau commutes over ZZ[q,q^-1]: {left * right == right * left}")
    print(f"Noncommuting Artin witness exists: {witness is not None}")
    return 0


def cmd_verify_b6() -> int:
    left = burau_word_matrix(6, bigelow_b6_left_twist_word(), ZZ)
    right = burau_word_matrix(6, bigelow_b6_right_twist_word(), ZZ)
    witness = first_noncommuting_generator_image(6, bigelow_b6_left_twist_word(), bigelow_b6_right_twist_word())
    print(f"Burau commutes over ZZ[q,q^-1]: {left * right == right * left}")
    print(f"Noncommuting Artin witness exists: {witness is not None}")
    return 0


def cmd_show_b4_q2() -> int:
    polynomial = bigelow_b4_q2_polynomial()
    print(polynomial)
    print(f"P(2) = {polynomial.evaluate(2)}")
    print(f"P(1) = {polynomial.evaluate(1)}")
    return 0


def cmd_orbit_n6() -> int:
    orbit = orbit_search_states(6, ZZ, WitnessType.PUNCTURE_PUNCTURE, 3, len(bigelow_b6_left_conjugator()))
    print(f"orbit size: {len(orbit)}")
    print(f"published left witness present: {burau_word_matrix(6, bigelow_b6_left_twist_word(), ZZ) in orbit}")
    return 0


def cmd_search_n6(mod_p: int | None) -> int:
    ring = ZZ if mod_p is None else FiniteFieldPrime(mod_p)
    config = SearchConfig(
        n=6,
        ring=ring,
        left_witness_type=WitnessType.PUNCTURE_PUNCTURE,
        left_base_index=3,
        left_max_depth=len(bigelow_b6_left_conjugator()),
        right_witness_type=WitnessType.PUNCTURE_PUNCTURE,
        right_base_index=3,
        right_max_depth=len(bigelow_b6_right_conjugator()),
    )
    result = find_commuting_kernel_pair(config)
    if result is None:
        print("No verified commuting pair found at the requested bounds.")
        return 1
    print(f"left conjugator depth: {result.left_state.depth}")
    print(f"right conjugator depth: {result.right_state.depth}")
    print(f"left conjugator: {result.left_state.conjugator}")
    print(f"right conjugator: {result.right_state.conjugator}")
    print(f"nontrivial generator witness: x_{result.nontrivial_generator}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "verify-b5":
        return cmd_verify_b5()
    if args.command == "verify-b6":
        return cmd_verify_b6()
    if args.command == "show-b4-q2":
        return cmd_show_b4_q2()
    if args.command == "orbit-n6":
        return cmd_orbit_n6()
    if args.command == "search-n6":
        return cmd_search_n6(args.mod_p)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
