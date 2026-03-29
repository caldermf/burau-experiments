from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch

from b5r2_dual_tables import build_b5r2_dual_table_data
from b5r2_hecke import (
    DIMENSION,
    evaluate_inverse_word,
    evaluate_word,
    matrix_support_bounds,
    commutator_is_identity,
    shift_matrix,
    sigma_matrix,
)


ROWS = DIMENSION * DIMENSION


@dataclass(frozen=True)
class CompiledOperator:
    row_out: torch.Tensor
    row_in: torch.Tensor
    shifts: torch.Tensor
    coeffs: torch.Tensor
    min_shift: int
    max_shift: int


@dataclass(frozen=True)
class CompiledOperatorTable:
    row_out: torch.Tensor
    row_in: torch.Tensor
    shifts: torch.Tensor
    coeffs: torch.Tensor
    valid_terms: torch.Tensor
    min_shift: int
    max_shift: int


@dataclass
class SearchConfig:
    bucket_cap: int = 20_000
    total_cap: int = 200_000
    max_depth: int = 40
    modulus: int = 5
    device: str = "auto"
    seed: int = 0
    max_witnesses: Optional[int] = 10
    print_witness_limit: int = 20
    witness_callback: Optional[Callable[[int, List[int], int], None]] = None
    exact_commutator_generator: Optional[int] = 1
    require_sigma2: bool = True


@dataclass
class Bucket:
    forward_states: torch.Tensor
    inverse_states: torch.Tensor
    last_simple_ids: torch.Tensor
    parent_indices: torch.Tensor
    used_sigma2: torch.Tensor
    record_indices: Optional[torch.Tensor] = None

    @property
    def size(self) -> int:
        return int(self.last_simple_ids.numel())


@dataclass
class SearchResult:
    witness: Optional[List[int]]
    depth: Optional[int]
    runtime_seconds: float
    device: str
    found: bool
    spread_counts_by_depth: Dict[int, Dict[int, int]]
    witnesses: List[List[int]]
    witness_depths: List[int]
    total_hits: int
    accepted_hits: int
    rejected_hits: int


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _flatten_index(row: int, col: int) -> int:
    return row * DIMENSION + col


def compile_right_matrix_operator(matrix) -> CompiledOperator:
    op: dict[tuple[int, int, int], int] = {}
    for out_row in range(DIMENSION):
        for out_col in range(DIMENSION):
            flat_out = _flatten_index(out_row, out_col)
            for mid in range(DIMENSION):
                flat_in = _flatten_index(out_row, mid)
                for shift, coeff in matrix[mid][out_col].items():
                    if coeff != 0:
                        op[(flat_out, flat_in, shift)] = op.get((flat_out, flat_in, shift), 0) + coeff
    return compile_sparse_operator(op)


def compile_left_matrix_operator(matrix) -> CompiledOperator:
    op: dict[tuple[int, int, int], int] = {}
    for out_row in range(DIMENSION):
        for out_col in range(DIMENSION):
            flat_out = _flatten_index(out_row, out_col)
            for mid in range(DIMENSION):
                flat_in = _flatten_index(mid, out_col)
                for shift, coeff in matrix[out_row][mid].items():
                    if coeff != 0:
                        op[(flat_out, flat_in, shift)] = op.get((flat_out, flat_in, shift), 0) + coeff
    return compile_sparse_operator(op)


def compile_sparse_operator(op: dict[tuple[int, int, int], int]) -> CompiledOperator:
    items = sorted(op.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    shifts = [key[2] for key, coeff in items if coeff != 0]
    min_shift = min(shifts) if shifts else 0
    max_shift = max(shifts) if shifts else 0
    row_out = torch.tensor([key[0] for key, _ in items], dtype=torch.long)
    row_in = torch.tensor([key[1] for key, _ in items], dtype=torch.long)
    shift_tensor = torch.tensor([key[2] for key, _ in items], dtype=torch.long)
    coeffs = torch.tensor([coeff for _, coeff in items], dtype=torch.int32)
    return CompiledOperator(
        row_out=row_out,
        row_in=row_in,
        shifts=shift_tensor,
        coeffs=coeffs,
        min_shift=min_shift,
        max_shift=max_shift,
    )


def stack_compiled_operators(operators: List[CompiledOperator], device: torch.device) -> CompiledOperatorTable:
    if not operators:
        empty_long = torch.empty((0, 0), dtype=torch.long, device=device)
        empty_int = torch.empty((0, 0), dtype=torch.int32, device=device)
        empty_bool = torch.empty((0, 0), dtype=torch.bool, device=device)
        return CompiledOperatorTable(
            row_out=empty_long,
            row_in=empty_long,
            shifts=empty_long,
            coeffs=empty_int,
            valid_terms=empty_bool,
            min_shift=0,
            max_shift=0,
        )

    max_terms = max(int(operator.coeffs.numel()) for operator in operators)
    count = len(operators)
    row_out = torch.zeros((count, max_terms), dtype=torch.long, device=device)
    row_in = torch.zeros((count, max_terms), dtype=torch.long, device=device)
    shifts = torch.zeros((count, max_terms), dtype=torch.long, device=device)
    coeffs = torch.zeros((count, max_terms), dtype=torch.int32, device=device)
    valid_terms = torch.zeros((count, max_terms), dtype=torch.bool, device=device)

    for index, operator in enumerate(operators):
        term_count = int(operator.coeffs.numel())
        if term_count == 0:
            continue
        row_out[index, :term_count] = operator.row_out.to(device)
        row_in[index, :term_count] = operator.row_in.to(device)
        shifts[index, :term_count] = operator.shifts.to(device)
        coeffs[index, :term_count] = operator.coeffs.to(device)
        valid_terms[index, :term_count] = True

    return CompiledOperatorTable(
        row_out=row_out,
        row_in=row_in,
        shifts=shifts,
        coeffs=coeffs,
        valid_terms=valid_terms,
        min_shift=min(operator.min_shift for operator in operators),
        max_shift=max(operator.max_shift for operator in operators),
    )


def _shift_rows(rows: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return rows
    width = rows.shape[-1]
    out = torch.zeros_like(rows)
    if shift > 0:
        if shift < width:
            out[..., shift:] = rows[..., : width - shift]
    else:
        offset = -shift
        if offset < width:
            out[..., : width - offset] = rows[..., offset:]
    return out


def apply_operator(states: torch.Tensor, operator: CompiledOperator, modulus: int) -> torch.Tensor:
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


def apply_operator_ids(states: torch.Tensor, operator_table: CompiledOperatorTable, operator_ids: torch.Tensor, modulus: int) -> torch.Tensor:
    if operator_ids.numel() == 0:
        return torch.empty((0, states.shape[0], ROWS, states.shape[-1]), dtype=states.dtype, device=states.device)

    selected_row_out = operator_table.row_out.index_select(0, operator_ids)
    selected_row_in = operator_table.row_in.index_select(0, operator_ids)
    selected_shifts = operator_table.shifts.index_select(0, operator_ids)
    selected_coeffs = operator_table.coeffs.index_select(0, operator_ids)
    selected_valid_terms = operator_table.valid_terms.index_select(0, operator_ids)

    batch_size, rows, width = states.shape
    num_operators, max_terms = selected_coeffs.shape
    out = torch.zeros((batch_size, num_operators, rows, width), dtype=states.dtype, device=states.device)
    target_rows = selected_row_out.view(1, num_operators, max_terms, 1).expand(batch_size, -1, -1, width)
    flat_row_in = selected_row_in.reshape(-1)

    for shift in range(operator_table.min_shift, operator_table.max_shift + 1):
        term_mask = selected_valid_terms & (selected_shifts == shift)
        if not term_mask.any():
            continue
        shifted = _shift_rows(states, shift)
        gathered = shifted[:, flat_row_in, :].view(batch_size, num_operators, max_terms, width)
        weights = (selected_coeffs * term_mask.to(selected_coeffs.dtype)).to(states.dtype)
        out.scatter_add_(2, target_rows, gathered * weights.view(1, num_operators, max_terms, 1))

    out = out.permute(1, 0, 2, 3).contiguous()
    if modulus > 0:
        out = torch.remainder(out, modulus)
    return out


def support_bounds(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if modulus > 0:
        states = torch.remainder(states, modulus)
    valuation, _, spread = support_bounds(states)
    normalized = torch.zeros_like(states)
    for shift_value in torch.unique(valuation):
        shift = -int(shift_value.item())
        mask = valuation == shift_value
        normalized[mask] = _shift_rows(states[mask], shift)
    if modulus > 0:
        normalized = torch.remainder(normalized, modulus)
    return normalized, spread


def make_identity_state(width: int, device: torch.device) -> torch.Tensor:
    state = torch.zeros((1, ROWS, width), dtype=torch.int32, device=device)
    for index in range(DIMENSION):
        state[0, _flatten_index(index, index), 0] = 1
    return state


def _priority_cap(bucket: Bucket, cap: int, generator: torch.Generator) -> Bucket:
    if bucket.size <= cap:
        return bucket
    priorities = torch.rand(bucket.size, generator=generator, device=bucket.forward_states.device)
    keep = torch.topk(priorities, k=cap, largest=False).indices
    return Bucket(
        forward_states=bucket.forward_states.index_select(0, keep),
        inverse_states=bucket.inverse_states.index_select(0, keep),
        last_simple_ids=bucket.last_simple_ids.index_select(0, keep),
        parent_indices=bucket.parent_indices.index_select(0, keep),
        used_sigma2=bucket.used_sigma2.index_select(0, keep),
    )


def _concat_bucket_parts(parts: list[Bucket]) -> Bucket:
    if len(parts) == 1:
        return parts[0]
    return Bucket(
        forward_states=torch.cat([part.forward_states for part in parts], dim=0),
        inverse_states=torch.cat([part.inverse_states for part in parts], dim=0),
        last_simple_ids=torch.cat([part.last_simple_ids for part in parts], dim=0),
        parent_indices=torch.cat([part.parent_indices for part in parts], dim=0),
        used_sigma2=torch.cat([part.used_sigma2 for part in parts], dim=0),
    )


def _selected_prev_spreads(buckets: Dict[int, Bucket], total_cap: int) -> List[int]:
    spreads = sorted(buckets.keys())
    selected: List[int] = []
    total = 0
    for spread in spreads:
        selected.append(spread)
        total += buckets[spread].size
        if total >= total_cap:
            break
    return selected


def _register_depth(
    depth: int,
    buckets: Dict[int, Bucket],
    history_last_simple_ids: List[Optional[torch.Tensor]],
    history_parent_indices: List[Optional[torch.Tensor]],
) -> None:
    spreads = sorted(buckets.keys())
    total = sum(bucket.size for bucket in buckets.values())
    last_simple_ids = torch.empty(total, dtype=torch.long)
    parent_indices = torch.empty(total, dtype=torch.long)
    offset = 0
    for spread in spreads:
        bucket = buckets[spread]
        count = bucket.size
        bucket.record_indices = torch.arange(offset, offset + count, dtype=torch.long, device=bucket.forward_states.device)
        last_simple_ids[offset : offset + count] = bucket.last_simple_ids.detach().cpu()
        parent_indices[offset : offset + count] = bucket.parent_indices.detach().cpu()
        offset += count
    while len(history_last_simple_ids) <= depth:
        history_last_simple_ids.append(None)
        history_parent_indices.append(None)
    history_last_simple_ids[depth] = last_simple_ids
    history_parent_indices[depth] = parent_indices


def _reconstruct_simple_id_path(
    history_last_simple_ids: List[Optional[torch.Tensor]],
    history_parent_indices: List[Optional[torch.Tensor]],
    depth: int,
    record_index: int,
) -> List[int]:
    out: List[int] = []
    current_depth = depth
    current_index = record_index
    while current_depth >= 1 and current_index >= 0:
        out.append(int(history_last_simple_ids[current_depth][current_index].item()))
        current_index = int(history_parent_indices[current_depth][current_index].item())
        current_depth -= 1
    return out


def _flatten_simple_path(simple_words: Sequence[Sequence[int]], simple_ids: Sequence[int]) -> List[int]:
    out: List[int] = []
    for simple_id in simple_ids:
        out.extend(simple_words[simple_id])
    return out


def pairwise_multiply_states(left: torch.Tensor, right: torch.Tensor, modulus: int, out_width: int) -> torch.Tensor:
    batch = left.shape[0]
    left_m = left.view(batch, DIMENSION, DIMENSION, left.shape[-1]).to(torch.int64)
    right_m = right.view(batch, DIMENSION, DIMENSION, right.shape[-1]).to(torch.int64)
    out = torch.zeros((batch, DIMENSION, DIMENSION, out_width), dtype=torch.int64, device=left.device)

    for mid in range(DIMENSION):
        left_slice = left_m[:, :, mid, :]
        right_slice = right_m[:, mid, :, :]
        for shift in range(right.shape[-1]):
            coeff = right_slice[:, :, shift]
            if torch.count_nonzero(coeff) == 0:
                continue
            limit = min(left.shape[-1], out_width - shift)
            if limit <= 0:
                break
            contribution = left_slice[:, :, :limit].unsqueeze(2) * coeff.unsqueeze(1).unsqueeze(3)
            out[:, :, :, shift : shift + limit] += contribution

    out = out.view(batch, ROWS, out_width).to(torch.int32)
    if modulus > 0:
        out = torch.remainder(out, modulus)
    return out


def _commutator_spreads(
    forward_states: torch.Tensor,
    inverse_states: torch.Tensor,
    sigma_left: CompiledOperator,
    sigma_inv_right: CompiledOperator,
    modulus: int,
    comm_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    temp = apply_operator(forward_states, sigma_inv_right, modulus)
    temp = apply_operator(temp, sigma_left, modulus)
    comm = pairwise_multiply_states(temp, inverse_states, modulus, out_width=comm_width)
    _, _, spreads = support_bounds(comm)
    return comm, spreads


def _projective_inverse_matrix(word: Sequence[int]):
    matrix = evaluate_inverse_word(word)
    bottom, _, _ = matrix_support_bounds(matrix)
    if bottom < 0:
        matrix = shift_matrix(matrix, -bottom)
    return matrix


def _build_initial_buckets(
    config: SearchConfig,
    simple_words: Sequence[Sequence[int]],
    start_simple_ids: Sequence[int],
    simple_uses_sigma2: torch.Tensor,
    forward_table: CompiledOperatorTable,
    inverse_table: CompiledOperatorTable,
    sigma_left: CompiledOperator,
    sigma_inv_right: CompiledOperator,
    state_width: int,
    comm_width: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Dict[int, Bucket], List[List[int]], List[int], int, int, int]:
    if not start_simple_ids:
        return {}, [], [], 0, 0, 0

    identity = make_identity_state(state_width, device=device)
    start_ids = torch.tensor(start_simple_ids, dtype=torch.long, device=device)
    start_forward = apply_operator_ids(identity, forward_table, start_ids, config.modulus).squeeze(1)
    start_inverse = apply_operator_ids(identity, inverse_table, start_ids, config.modulus).squeeze(1)
    start_forward, _ = normalize_states(start_forward, config.modulus)
    start_inverse, _ = normalize_states(start_inverse, config.modulus)
    _, start_spreads = _commutator_spreads(start_forward, start_inverse, sigma_left, sigma_inv_right, config.modulus, comm_width)

    witnesses: List[List[int]] = []
    witness_depths: List[int] = []
    total_hits = 0
    accepted_hits = 0
    rejected_hits = 0
    parts: dict[int, list[Bucket]] = {}

    for index, simple_id in enumerate(start_simple_ids):
        spread = int(start_spreads[index].item())
        if spread < 0:
            continue
        if spread == 0:
            total_hits += 1
            candidate = list(simple_words[simple_id])
            accept = True
            if config.require_sigma2 and not bool(simple_uses_sigma2[simple_id].item()):
                accept = False
            if accept and config.exact_commutator_generator is not None:
                accept = commutator_is_identity(candidate, modulus=config.modulus, generator=config.exact_commutator_generator)
            if accept:
                accepted_hits += 1
                witnesses.append(candidate)
                witness_depths.append(1)
            else:
                rejected_hits += 1
            continue

        parts.setdefault(spread, []).append(
            Bucket(
                forward_states=start_forward[index : index + 1],
                inverse_states=start_inverse[index : index + 1],
                last_simple_ids=torch.tensor([simple_id], dtype=torch.long, device=device),
                parent_indices=torch.tensor([-1], dtype=torch.long, device=device),
                used_sigma2=simple_uses_sigma2[start_ids[index : index + 1]],
            )
        )

    buckets = {spread: _priority_cap(_concat_bucket_parts(group), config.bucket_cap, generator) for spread, group in parts.items()}
    return buckets, witnesses, witness_depths, total_hits, accepted_hits, rejected_hits


def run_search(config: SearchConfig) -> SearchResult:
    if config.modulus <= 0:
        raise ValueError("modulus must be positive")
    if config.max_witnesses is not None and config.max_witnesses < 0:
        raise ValueError("max_witnesses must be non-negative or None")
    if config.print_witness_limit < 0:
        raise ValueError("print_witness_limit must be non-negative")
    if config.exact_commutator_generator not in (None, 1):
        raise ValueError("The B5 dual-simple search currently supports only commutators with sigma_1.")

    device = _resolve_device(config.device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(config.seed)

    tables = build_b5r2_dual_table_data(device=device)
    simple_forward_ops = [compile_left_matrix_operator(evaluate_word(word)) for word in tables.simple_words]
    simple_inverse_ops = [compile_right_matrix_operator(_projective_inverse_matrix(word)) for word in tables.simple_words]
    forward_table = stack_compiled_operators(simple_forward_ops, device=device)
    inverse_table = stack_compiled_operators(simple_inverse_ops, device=device)

    sigma_left = compile_left_matrix_operator(sigma_matrix(1))
    sigma_inv_right = compile_right_matrix_operator(_projective_inverse_matrix([1]))
    sigma_left = CompiledOperator(
        row_out=sigma_left.row_out.to(device),
        row_in=sigma_left.row_in.to(device),
        shifts=sigma_left.shifts.to(device),
        coeffs=sigma_left.coeffs.to(device),
        min_shift=sigma_left.min_shift,
        max_shift=sigma_left.max_shift,
    )
    sigma_inv_right = CompiledOperator(
        row_out=sigma_inv_right.row_out.to(device),
        row_in=sigma_inv_right.row_in.to(device),
        shifts=sigma_inv_right.shifts.to(device),
        coeffs=sigma_inv_right.coeffs.to(device),
        min_shift=sigma_inv_right.min_shift,
        max_shift=sigma_inv_right.max_shift,
    )

    forward_growth = max((operator.max_shift for operator in simple_forward_ops), default=0)
    inverse_growth = max((operator.max_shift for operator in simple_inverse_ops), default=0)
    state_width = max(config.max_depth * max(forward_growth, inverse_growth) + 8, 8)
    comm_width = 2 * state_width + sigma_left.max_shift + sigma_inv_right.max_shift + 4

    simple_uses_sigma2 = torch.tensor(
        [any(abs(letter) == 2 for letter in word) for word in tables.simple_words],
        dtype=torch.bool,
        device=device,
    )

    history_last_simple_ids: List[Optional[torch.Tensor]] = [None]
    history_parent_indices: List[Optional[torch.Tensor]] = [None]

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start_event is not None:
        start_event.record()
    else:
        import time

        wall_start = time.time()

    print(
        "Starting B5 (3,2) Hecke dual-simple reservoir search with p=",
        config.modulus,
        "bucket_cap",
        config.bucket_cap,
        "total_cap",
        config.total_cap,
        "max_depth",
        config.max_depth,
        "device",
        device,
    )
    print(f"We have {len(tables.atom_words)} atoms and {len(tables.simple_words)} dual simples")
    print(f"Admissible start simples: {len(tables.start_simple_ids)}")

    current_buckets, witnesses, witness_depths, total_hits, accepted_hits, rejected_hits = _build_initial_buckets(
        config=config,
        simple_words=tables.simple_words,
        start_simple_ids=tables.start_simple_ids,
        simple_uses_sigma2=simple_uses_sigma2,
        forward_table=forward_table,
        inverse_table=inverse_table,
        sigma_left=sigma_left,
        sigma_inv_right=sigma_inv_right,
        state_width=state_width,
        comm_width=comm_width,
        device=device,
        generator=generator,
    )

    witness = witnesses[0] if witnesses else None
    witness_depth = witness_depths[0] if witness_depths else None
    seen_witnesses = {tuple(candidate) for candidate in witnesses}

    if current_buckets:
        _register_depth(1, current_buckets, history_last_simple_ids, history_parent_indices)
        spread_counts_by_depth: Dict[int, Dict[int, int]] = {
            1: {spread: bucket.size for spread, bucket in sorted(current_buckets.items())}
        }
    else:
        spread_counts_by_depth = {1: {}}
    print("Initialized step 1 with spreads:", spread_counts_by_depth[1])

    if witness is not None and config.witness_callback is not None:
        config.witness_callback(1, witness, 1)
    if witness is not None and config.print_witness_limit > 0:
        print(f"Found witness #1 at depth 1: {witness}")

    stop_after_hit = config.max_witnesses is not None and accepted_hits >= config.max_witnesses

    with torch.no_grad():
        for depth in range(2, config.max_depth + 1):
            if stop_after_hit or not current_buckets:
                break

            selected_spreads = _selected_prev_spreads(current_buckets, config.total_cap)
            if not selected_spreads:
                break

            print(f"Starting step {depth}")
            next_parts: dict[int, list[Bucket]] = {}
            total_drops = 0

            for prev_spread in selected_spreads:
                bucket = current_buckets[prev_spread]
                print(f"Analyzing {bucket.size} states in spread {prev_spread}")
                unique_right_ids = torch.unique(bucket.last_simple_ids)

                for right_id_tensor in unique_right_ids:
                    right_id = int(right_id_tensor.item())
                    mask = bucket.last_simple_ids == right_id_tensor
                    group_forward = bucket.forward_states[mask]
                    group_inverse = bucket.inverse_states[mask]
                    group_parents = bucket.record_indices[mask]
                    group_used_sigma2 = bucket.used_sigma2[mask]

                    predecessor_count = int(tables.allowed_count[right_id].item())
                    if predecessor_count == 0:
                        continue
                    predecessor_ids = tables.allowed_predecessors_padded[right_id, :predecessor_count]

                    child_forward = apply_operator_ids(group_forward, forward_table, predecessor_ids, config.modulus)
                    child_inverse = apply_operator_ids(group_inverse, inverse_table, predecessor_ids, config.modulus)
                    num_predecessors, group_size = child_forward.shape[:2]
                    child_forward = child_forward.view(num_predecessors * group_size, *child_forward.shape[2:])
                    child_inverse = child_inverse.view(num_predecessors * group_size, *child_inverse.shape[2:])
                    child_forward, _ = normalize_states(child_forward, config.modulus)
                    child_inverse, _ = normalize_states(child_inverse, config.modulus)
                    _, spreads = _commutator_spreads(
                        child_forward,
                        child_inverse,
                        sigma_left,
                        sigma_inv_right,
                        config.modulus,
                        comm_width,
                    )

                    valid = spreads >= 0
                    if not valid.any():
                        continue

                    flat_predecessor_ids = predecessor_ids.unsqueeze(1).expand(-1, group_size).reshape(-1)
                    flat_parents = group_parents.unsqueeze(0).expand(num_predecessors, -1).reshape(-1)
                    flat_used_sigma2 = (
                        group_used_sigma2.unsqueeze(0).expand(num_predecessors, -1).reshape(-1)
                        | simple_uses_sigma2.index_select(0, flat_predecessor_ids)
                    )

                    drops_mask = valid & (spreads < prev_spread)
                    total_drops += int(drops_mask.sum().item())

                    hit_mask = valid & (spreads == 0)
                    if hit_mask.any():
                        hit_indices = torch.nonzero(hit_mask, as_tuple=False).squeeze(1)
                        total_hits += int(hit_indices.numel())
                        for hit_idx in hit_indices.tolist():
                            predecessor_id = int(flat_predecessor_ids[hit_idx].item())
                            parent_record = int(flat_parents[hit_idx].item())
                            simple_path = [predecessor_id] + _reconstruct_simple_id_path(
                                history_last_simple_ids,
                                history_parent_indices,
                                depth - 1,
                                parent_record,
                            )
                            candidate = _flatten_simple_path(tables.simple_words, simple_path)
                            key = tuple(candidate)
                            if key in seen_witnesses:
                                continue
                            seen_witnesses.add(key)

                            accept = True
                            if config.require_sigma2 and not bool(flat_used_sigma2[hit_idx].item()):
                                accept = False
                            if accept and config.exact_commutator_generator is not None:
                                accept = commutator_is_identity(
                                    candidate,
                                    modulus=config.modulus,
                                    generator=config.exact_commutator_generator,
                                )
                            if accept:
                                accepted_hits += 1
                                witnesses.append(candidate)
                                witness_depths.append(depth)
                                if witness is None:
                                    witness = candidate
                                    witness_depth = depth
                                if len(witnesses) <= config.print_witness_limit:
                                    print(f"Found witness #{len(witnesses)} at depth {depth}: {candidate}")
                                elif len(witnesses) == config.print_witness_limit + 1:
                                    print(
                                        f"Reached witness print limit ({config.print_witness_limit}); suppressing further witness dumps."
                                    )
                                if config.witness_callback is not None:
                                    config.witness_callback(depth, candidate, accepted_hits)
                                if config.max_witnesses is not None and accepted_hits >= config.max_witnesses:
                                    stop_after_hit = True
                                    break
                            else:
                                rejected_hits += 1
                        if stop_after_hit:
                            break

                    valid &= spreads != 0
                    if not valid.any():
                        continue

                    kept = torch.nonzero(valid, as_tuple=False).squeeze(1)
                    kept_spreads = spreads.index_select(0, kept)
                    kept_forward = child_forward.index_select(0, kept)
                    kept_inverse = child_inverse.index_select(0, kept)
                    kept_parents = flat_parents.index_select(0, kept)
                    kept_used_sigma2 = flat_used_sigma2.index_select(0, kept)
                    kept_simple_ids = flat_predecessor_ids.index_select(0, kept)

                    for spread_value in torch.unique(kept_spreads):
                        spread_int = int(spread_value.item())
                        spread_mask = kept_spreads == spread_value
                        next_parts.setdefault(spread_int, []).append(
                            Bucket(
                                forward_states=kept_forward[spread_mask],
                                inverse_states=kept_inverse[spread_mask],
                                last_simple_ids=kept_simple_ids[spread_mask],
                                parent_indices=kept_parents[spread_mask],
                                used_sigma2=kept_used_sigma2[spread_mask],
                            )
                        )

                if stop_after_hit:
                    break

            if stop_after_hit:
                break

            current_buckets = {
                spread: _priority_cap(_concat_bucket_parts(parts), config.bucket_cap, generator)
                for spread, parts in next_parts.items()
            }
            if not current_buckets:
                spread_counts_by_depth[depth] = {}
                break

            _register_depth(depth, current_buckets, history_last_simple_ids, history_parent_indices)
            counts = {spread: bucket.size for spread, bucket in sorted(current_buckets.items())}
            spread_counts_by_depth[depth] = counts
            print(f"Finished step {depth}, minimal spread {min(counts)}, max spread {max(counts)}, got {total_drops} drops")
            print(f"Total number of states {sum(counts.values())}")

    if end_event is not None:
        end_event.record()
        torch.cuda.synchronize(device)
        runtime_seconds = start_event.elapsed_time(end_event) / 1000.0
    else:
        runtime_seconds = time.time() - wall_start

    return SearchResult(
        witness=witness,
        depth=witness_depth,
        runtime_seconds=runtime_seconds,
        device=str(device),
        found=witness is not None,
        spread_counts_by_depth=spread_counts_by_depth,
        witnesses=witnesses,
        witness_depths=witness_depths,
        total_hits=total_hits,
        accepted_hits=accepted_hits,
        rejected_hits=rejected_hits,
    )
