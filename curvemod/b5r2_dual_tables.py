from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import torch


BraidWord = List[int]


def _atom(i: int, j: int) -> BraidWord:
    word = list(range(i, j))
    word.extend(-k for k in range(j - 2, i - 1, -1))
    return word


def _free_reduce(word: Sequence[int]) -> tuple[int, ...]:
    out: list[int] = []
    for letter in word:
        if out and out[-1] == -letter:
            out.pop()
        else:
            out.append(letter)
    return tuple(out)


def _free_mul(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return _free_reduce(tuple(left) + tuple(right))


def _free_inv(word: Sequence[int]) -> tuple[int, ...]:
    return tuple(-letter for letter in reversed(word))


def _substitute(word: Sequence[int], images: Dict[int, tuple[int, ...]]) -> tuple[int, ...]:
    out: tuple[int, ...] = ()
    for letter in word:
        piece = images[abs(letter)]
        if letter < 0:
            piece = _free_inv(piece)
        out = _free_mul(out, piece)
    return out


def _braid_automorphism(letter: int, n: int = 5) -> Dict[int, tuple[int, ...]]:
    index = abs(letter)
    images = {k: (k,) for k in range(1, n + 1)}
    if letter > 0:
        images[index] = _free_reduce((index, index + 1, -index))
        images[index + 1] = (index,)
    else:
        images[index] = (index + 1,)
        images[index + 1] = _free_reduce((-(index + 1), index, index + 1))
    return images


def _compose_automorphisms(
    left: Dict[int, tuple[int, ...]],
    right: Dict[int, tuple[int, ...]],
    n: int = 5,
) -> Dict[int, tuple[int, ...]]:
    return {k: _substitute(right[k], left) for k in range(1, n + 1)}


def _eval_braid(word: Sequence[int], n: int = 5) -> tuple[tuple[int, ...], ...]:
    automorphism = {k: (k,) for k in range(1, n + 1)}
    for letter in word:
        automorphism = _compose_automorphisms(_braid_automorphism(letter, n), automorphism, n)
    return tuple(automorphism[k] for k in range(1, n + 1))


def _set_partitions(seq: list[int]):
    if not seq:
        yield []
        return
    first, *rest = seq
    for partition in _set_partitions(rest):
        yield [[first]] + [block[:] for block in partition]
        for index in range(len(partition)):
            new_partition = [block[:] for block in partition]
            new_partition[index] = new_partition[index] + [first]
            yield new_partition


def _canonical_partition(partition: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    blocks = [tuple(sorted(block)) for block in partition]
    return tuple(sorted(blocks, key=lambda block: (min(block), len(block), block)))


def _is_noncrossing(partition: Sequence[Sequence[int]]) -> bool:
    blocks = [tuple(sorted(block)) for block in partition]
    for left_index, left_block in enumerate(blocks):
        for right_block in blocks[left_index + 1 :]:
            for a, c in combinations(left_block, 2):
                for b, d in combinations(right_block, 2):
                    if a < b < c < d or b < a < d < c:
                        return False
    return True


def _block_word(block: Sequence[int]) -> BraidWord:
    if len(block) <= 1:
        return []
    sorted_block = sorted(block)
    out: BraidWord = []
    for left, right in zip(sorted_block, sorted_block[1:]):
        out.extend(_atom(left, right))
    return out


@dataclass
class B5R2DualTableData:
    simple_words: List[BraidWord]
    atom_words: List[BraidWord]
    sigma1_atom_id: int
    gamma_id: int
    allowed_predecessors: List[List[int]]
    start_simple_ids: List[int]
    allowed_predecessors_padded: torch.Tensor
    allowed_count: torch.Tensor


@lru_cache(maxsize=1)
def _build_cached_data() -> tuple[List[BraidWord], List[BraidWord], int, int, List[List[int]], List[int]]:
    n = 5
    atom_words = [_atom(i, j) for i in range(1, n) for j in range(i + 1, n + 1)]

    noncrossing_partitions: list[tuple[tuple[int, ...], ...]] = []
    for partition in _set_partitions(list(range(1, n + 1))):
        canonical = _canonical_partition(partition)
        if canonical in noncrossing_partitions:
            continue
        if _is_noncrossing(canonical):
            noncrossing_partitions.append(canonical)
    noncrossing_partitions.sort()

    simple_words: list[BraidWord] = []
    key_to_simple_id: dict[tuple[tuple[int, ...], ...], int] = {}
    for partition in noncrossing_partitions:
        word: BraidWord = []
        for block in partition:
            word.extend(_block_word(block))
        key = _eval_braid(word, n)
        key_to_simple_id[key] = len(simple_words)
        simple_words.append(word)

    identity_id = key_to_simple_id[_eval_braid([], n)]
    sigma1_atom_id = key_to_simple_id[_eval_braid([1], n)]
    gamma_id = key_to_simple_id[_eval_braid([1, 2, 3, 4], n)]

    def is_simple(word: Sequence[int]) -> bool:
        return _eval_braid(word, n) in key_to_simple_id

    right_divisors: list[set[tuple[int, ...]]] = [set() for _ in simple_words]
    for prefix_word in simple_words:
        for atom in atom_words:
            product_key = _eval_braid(prefix_word + atom, n)
            simple_id = key_to_simple_id.get(product_key)
            if simple_id is not None:
                right_divisors[simple_id].add(tuple(atom))
    right_divisors_lists = [sorted(items) for items in right_divisors]

    allowed_predecessors: list[list[int]] = [[] for _ in simple_words]
    for right_id, right_word in enumerate(simple_words):
        for left_id, left_word in enumerate(simple_words):
            if left_id in (identity_id, gamma_id):
                continue
            admissible = True
            for descent in right_divisors_lists[left_id]:
                if is_simple(list(descent) + right_word):
                    admissible = False
                    break
            if admissible:
                allowed_predecessors[right_id].append(left_id)

    sigma1_word = [1]

    def sigma1_divides(simple_word: Sequence[int]) -> bool:
        target = _eval_braid(simple_word, n)
        for prefix in simple_words:
            if is_simple(prefix + sigma1_word) and _eval_braid(prefix + sigma1_word, n) == target:
                return True
        return False

    def normal_pair(left_word: Sequence[int], right_word: Sequence[int]) -> bool:
        left_id = key_to_simple_id[_eval_braid(left_word, n)]
        for descent in right_divisors_lists[left_id]:
            if is_simple(list(descent) + list(right_word)):
                return False
        return True

    start_simple_ids = [
        simple_id
        for simple_id, word in enumerate(simple_words)
        if simple_id not in (identity_id, gamma_id) and not sigma1_divides(word) and normal_pair(word, sigma1_word)
    ]

    return simple_words, atom_words, sigma1_atom_id, gamma_id, allowed_predecessors, start_simple_ids


def build_b5r2_dual_table_data(device: torch.device) -> B5R2DualTableData:
    simple_words, atom_words, sigma1_atom_id, gamma_id, allowed_predecessors, start_simple_ids = _build_cached_data()

    max_allowed = max((len(items) for items in allowed_predecessors), default=0)
    if max_allowed == 0:
        allowed_predecessors_padded = torch.empty((len(simple_words), 0), dtype=torch.long, device=device)
    else:
        allowed_predecessors_padded = torch.full((len(simple_words), max_allowed), -1, dtype=torch.long, device=device)
        for row, items in enumerate(allowed_predecessors):
            if items:
                allowed_predecessors_padded[row, : len(items)] = torch.tensor(items, dtype=torch.long, device=device)

    allowed_count = torch.tensor([len(items) for items in allowed_predecessors], dtype=torch.long, device=device)

    return B5R2DualTableData(
        simple_words=[word[:] for word in simple_words],
        atom_words=[word[:] for word in atom_words],
        sigma1_atom_id=sigma1_atom_id,
        gamma_id=gamma_id,
        allowed_predecessors=allowed_predecessors,
        start_simple_ids=start_simple_ids,
        allowed_predecessors_padded=allowed_predecessors_padded,
        allowed_count=allowed_count,
    )
