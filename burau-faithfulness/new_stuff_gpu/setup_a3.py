from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from burau_setup import laurent


# Type A3 with linear orientation 1 -> 2 -> 3.
positive_letters = [1, 2, 3]
negative_letters = [-1, -2, -3]
all_letters = positive_letters + negative_letters
restricted_letters = all_letters

coxeter_matrix = np.array(
    [
        [1, 3, 2],
        [3, 1, 3],
        [2, 3, 1],
    ]
)

dynkin_graph = {
    1: {2},
    2: {1, 3},
    3: {2},
}

dynkin_ograph = {
    1: {2},
    2: {3},
    3: set(),
}


def exists_dynkin_graph_edge(i, j):
    return j in dynkin_graph.get(i, set())


def exists_dynkin_ograph_edge(i, j):
    return j in dynkin_ograph.get(i, set())


a1 = np.array([1, 0, 0])
a2 = np.array([0, 1, 0])
a3 = np.array([0, 0, 1])
alpha = [a1, a2, a3]

ROOTS = [
    a1,
    a2,
    a3,
    a1 + a2,
    a2 + a3,
    a1 + a2 + a3,
]

# Dual BKL atoms for A3, written with the same sign convention as Bucket_D4.
# Stored words act by applying the list in reverse order.
DUAL_ATOMS = [
    [-1],
    [-2],
    [-3],
    [1, -2, -1],
    [2, -3, -2],
    [1, 2, -3, -2, -1],
]

# This stored word acts as sigma_1 sigma_2 sigma_3 after reversal.
GAMMA_WORD = [-3, -2, -1]


def makekey(root, nb):
    key_tuple = (root[0], root[1], root[2], nb)
    return "-".join(str(x) for x in key_tuple)


def make_burau_fn(i):
    i_sgn = -1 if i < 0 else 1
    index = abs(i) - 1

    def burau_fn_i(input_vec):
        updated_coordinate = {}
        for k in range(len(input_vec)):
            if k == index:
                kth_summand = laurent.product({-2 * i_sgn: -1}, input_vec[k])
            elif coxeter_matrix[index, k] > 2:
                kth_summand = laurent.product({-1 * i_sgn: -1}, input_vec[k])
            else:
                kth_summand = {}

            updated_coordinate = laurent.trim(laurent.addition(updated_coordinate, kth_summand))

        output_vec = input_vec.copy()
        output_vec[index] = updated_coordinate
        return output_vec

    return burau_fn_i


def make_oburau_fn(i):
    i_sgn = -1 if i < 0 else 1
    index = abs(i) - 1

    def oburau_fn_i(input_vec):
        updated_coordinate = {}
        for k in range(len(input_vec)):
            if k == index:
                kth_summand = laurent.product({-i_sgn: -1}, input_vec[k])
            elif coxeter_matrix[index, k] > 2:
                if exists_dynkin_ograph_edge(index + 1, k + 1):
                    kth_summand = laurent.product({0 if i > 0 else -1 * i_sgn: -1}, input_vec[k])
                elif exists_dynkin_ograph_edge(k + 1, index + 1):
                    kth_summand = laurent.product({-1 * i_sgn if i > 0 else 0: -1}, input_vec[k])
                else:
                    kth_summand = {}
            else:
                kth_summand = {}

            updated_coordinate = laurent.trim(laurent.addition(updated_coordinate, kth_summand))

        output_vec = input_vec.copy()
        output_vec[index] = updated_coordinate
        return output_vec

    return oburau_fn_i


def make_pi_dim_vector(i):
    output_vec = [{} for _ in range(len(dynkin_graph))]
    output_vec[i - 1] = {0: 1}
    return output_vec


burau_fns = {i: make_burau_fn(i) for i in all_letters}
oburau_fns = {i: make_oburau_fn(i) for i in all_letters}
dim_vectors = {i: make_pi_dim_vector(i) for i in positive_letters}


def pairing(dim_vec1, dim_vec2):
    total_sum = {}
    for i in range(len(dim_vec1)):
        fi = laurent.qtoqinv(dim_vec1[i])
        tmp_sum = {}
        for j in range(len(dim_vec2)):
            gj = dim_vec2[j]
            if j == i:
                tmp_sum = laurent.addition(tmp_sum, laurent.product({0: 1, 2: 1}, gj))
            elif exists_dynkin_graph_edge(i + 1, j + 1):
                tmp_sum = laurent.addition(tmp_sum, laurent.product({1: 1}, gj))
        total_sum = laurent.addition(total_sum, laurent.product(fi, tmp_sum))

    return laurent.trim(total_sum)


def opairing(dim_vec1, dim_vec2):
    total_sum = {}
    for i in range(len(dim_vec1)):
        fi = laurent.qtoqinv(dim_vec1[i])
        tmp_sum = {}
        for j in range(len(dim_vec2)):
            gj = dim_vec2[j]
            if j == i:
                tmp_sum = laurent.addition(tmp_sum, laurent.product({0: 1, 1: 1}, gj))
            elif exists_dynkin_graph_edge(i + 1, j + 1):
                if exists_dynkin_ograph_edge(i + 1, j + 1):
                    tmp_sum = laurent.addition(tmp_sum, laurent.product({1: 1}, gj))
                else:
                    tmp_sum = laurent.addition(tmp_sum, laurent.product({0: 1}, gj))
        total_sum = laurent.addition(total_sum, laurent.product(fi, tmp_sum))

    return laurent.trim(total_sum)


def poly_normalize_vector(dim_vec):
    bottom_deg = np.inf
    min_index = None

    for k, poly in enumerate(dim_vec):
        new_valuation = np.inf if poly == {} else min(poly.keys())
        if new_valuation < bottom_deg:
            bottom_deg = new_valuation
            min_index = k

    if min_index is None:
        return dim_vec

    leading_coeff_index = max(dim_vec[min_index].keys())
    leading_coeff = dim_vec[min_index][leading_coeff_index]
    sgn = 1 if leading_coeff >= 0 else -1
    q_factor = {-bottom_deg: sgn}
    return [laurent.product(q_factor, d) for d in dim_vec]


def find_ends_vector(dim_vec):
    dequantized_dim_vec = [laurent.dequantize(d) for d in dim_vec]
    return sum(abs(dequantized_dim_vec[i]) * alpha[i] for i in range(len(alpha)))


def nb_terms_vector(dim_vec):
    return sum(sum(abs(v) for v in d.values()) for d in dim_vec)


def topdeg_vector(dim_vec):
    return max(laurent.degree(x) for x in dim_vec)


def botdeg_vector(dim_vec):
    return min(laurent.valuation(x) for x in dim_vec)
