from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch

import setup_a3 as ct


OperatorDict = Dict[Tuple[int, int, int], int]


@dataclass(frozen=True)
class CompiledOperator:
    word: tuple[int, ...]
    row_out: torch.Tensor
    row_in: torch.Tensor
    shifts: torch.Tensor
    coeffs: torch.Tensor
    min_shift: int
    max_shift: int


def _identity_operator() -> OperatorDict:
    n = len(ct.positive_letters)
    return {(row, row, 0): 1 for row in range(n)}


def _trim_operator(op: OperatorDict) -> OperatorDict:
    return {key: value for key, value in op.items() if value != 0}


def make_oriented_letter_operator(letter: int) -> OperatorDict:
    """
    Return the oriented Burau action for a single Artin letter as a sparse
    Laurent-matrix operator encoded by (row_out, row_in, q_shift) -> coeff.
    """
    n = len(ct.positive_letters)
    sign = -1 if letter < 0 else 1
    index = abs(letter) - 1

    op: OperatorDict = {}
    for row in range(n):
        if row != index:
            op[(row, row, 0)] = 1
            continue

        for col in range(n):
            coeff = 0
            shift = 0
            if col == index:
                coeff = -1
                shift = -sign
            elif ct.coxeter_matrix[index, col] > 2:
                if ct.exists_dynkin_ograph_edge(index + 1, col + 1):
                    coeff = -1
                    shift = 0 if letter > 0 else 1
                elif ct.exists_dynkin_ograph_edge(col + 1, index + 1):
                    coeff = -1
                    shift = -1 if letter > 0 else 0
            if coeff != 0:
                op[(row, col, shift)] = coeff

    return op


def compose_operators(left: OperatorDict, right: OperatorDict) -> OperatorDict:
    """
    Compose two sparse Laurent-matrix operators: left(right(v)).
    """
    right_by_out: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for (mid_out, col, shift), coeff in right.items():
        right_by_out[mid_out].append((col, shift, coeff))

    out: dict[tuple[int, int, int], int] = defaultdict(int)
    for (row, mid, left_shift), left_coeff in left.items():
        for col, right_shift, right_coeff in right_by_out.get(mid, []):
            out[(row, col, left_shift + right_shift)] += left_coeff * right_coeff

    return _trim_operator(out)


def compile_word_operator(word: Iterable[int], device: torch.device) -> CompiledOperator:
    op = _identity_operator()
    word_tuple = tuple(word)
    for letter in reversed(word_tuple):
        op = compose_operators(make_oriented_letter_operator(letter), op)

    shifts = [shift for (_, _, shift), coeff in op.items() if coeff != 0]
    min_shift = min(shifts) if shifts else 0
    max_shift = max(shifts) if shifts else 0

    items = sorted(op.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    row_out = torch.tensor([key[0] for key, _ in items], dtype=torch.long, device=device)
    row_in = torch.tensor([key[1] for key, _ in items], dtype=torch.long, device=device)
    shift_tensor = torch.tensor([key[2] for key, _ in items], dtype=torch.long, device=device)
    coeffs = torch.tensor([coeff for _, coeff in items], dtype=torch.int32, device=device)

    return CompiledOperator(
        word=word_tuple,
        row_out=row_out,
        row_in=row_in,
        shifts=shift_tensor,
        coeffs=coeffs,
        min_shift=min_shift,
        max_shift=max_shift,
    )


def compile_simple_operators(words: List[Iterable[int]], device: torch.device) -> List[CompiledOperator]:
    return [compile_word_operator(word, device=device) for word in words]


def _shift_rows(rows: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shift the q-degree axis with zero fill rather than wraparound.
    """
    if shift == 0:
        return rows

    batch, width = rows.shape
    out = torch.zeros_like(rows)
    if shift > 0:
        if shift < width:
            out[:, shift:] = rows[:, : width - shift]
    else:
        offset = -shift
        if offset < width:
            out[:, : width - offset] = rows[:, offset:]
    return out


def apply_operator(states: torch.Tensor, operator: CompiledOperator, modulus: int) -> torch.Tensor:
    """
    Apply a compiled operator to a batch of normalized states.
    """
    out = torch.zeros_like(states)
    term_count = int(operator.coeffs.numel())
    for term_idx in range(term_count):
        row_out = int(operator.row_out[term_idx].item())
        row_in = int(operator.row_in[term_idx].item())
        shift = int(operator.shifts[term_idx].item())
        coeff = int(operator.coeffs[term_idx].item())
        shifted = _shift_rows(states[:, row_in, :], shift)
        if coeff == 1:
            out[:, row_out, :] += shifted
        elif coeff == -1:
            out[:, row_out, :] -= shifted
        else:
            out[:, row_out, :] += coeff * shifted

    if modulus > 0:
        out = torch.remainder(out, modulus)
    return out


def support_bounds(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return valuation, top degree, and spread for a batch of states.
    States are assumed to live in a fixed width q-window.
    """
    support = states.ne(0).any(dim=1)
    has_support = support.any(dim=1)
    first = support.to(torch.int64).argmax(dim=1)
    last = states.shape[-1] - 1 - support.flip(dims=[1]).to(torch.int64).argmax(dim=1)
    spread = last - first

    no_support = ~has_support
    if no_support.any():
        first = first.masked_fill(no_support, 0)
        last = last.masked_fill(no_support, -1)
        spread = spread.masked_fill(no_support, -1)

    return first, last, spread


def normalize_states(states: torch.Tensor, modulus: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize states by shifting valuation to zero; return normalized states and spread.
    """
    if modulus > 0:
        states = torch.remainder(states, modulus)

    valuation, _, spread = support_bounds(states)
    width = states.shape[-1]
    idx = torch.arange(width, device=states.device, dtype=torch.long).unsqueeze(0) + valuation.unsqueeze(1)
    valid = idx < width
    safe_idx = idx.clamp(max=width - 1)
    gather_idx = safe_idx.unsqueeze(1).expand(-1, states.shape[1], -1)
    normalized = torch.gather(states, 2, gather_idx)
    normalized = normalized * valid.unsqueeze(1).to(normalized.dtype)

    if modulus > 0:
        normalized = torch.remainder(normalized, modulus)
    return normalized, spread


def make_initial_state(width: int, device: torch.device) -> torch.Tensor:
    state = torch.zeros((1, len(ct.positive_letters), width), dtype=torch.int32, device=device)
    state[0, 0, 0] = 1
    return state
