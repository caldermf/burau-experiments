import numpy as np
import sys
sys.path.append("..")
from burau_setup.digraph_data import DigraphData

def test_k4_setup():
    digraph_k4 = {1: set([2,3,4]), 2: set([1,3,4]), 3: set([1, 2,4]), 4: set([1,2,3])}
    k4 = DigraphData(digraph_k4)

    assert k4.positive_letters == [1, 2, 3, 4]
    assert k4.negative_letters == [-1, -2, -3, -4]
    assert k4.all_letters == [1, 2, 3, 4, -1, -2, -3, -4]
    assert k4.restricted_letters == [1, 2, 3, 4, -1, -2, -3, -4]
    d1, d2 = k4.dim_vectors[1], k4.dim_vectors[2]
    assert k4.pairing(d1,d2) == {1:1}
    #assert k4.coxeter_matrix == np.array([[1, 3, 3, 3],[3, 1, 3, 3],[3, 3, 1, 3],[3, 3, 3, 1]])
    print("All tests for K4 passed!")
    return k4

def test_a3_setup():
    digraph_a3 = {1: set([2,3]), 2: set([1,3]), 3: set([1, 2])}
    a3 = DigraphData(digraph_a3)

    assert a3.positive_letters == [1, 2, 3]
    assert a3.negative_letters == [-1, -2, -3]
    assert a3.all_letters == [1, 2, 3, -1, -2, -3]
    assert a3.restricted_letters == [1, 2, 3, -1, -2, -3]
    return a3
    
