###########
# Imports #
###########
# Standard libraries
import time # for tracking
import json # library for storing data
import numpy as np
import os
import glob
import re

# Local packages
load("cobracat-master/zigzagalgebra_element.pyx")
load("cobracat-master/zigzagalgebra.sage")
load("cobracat-master/projective-zigzagmodules.sage")
load("cobracat-master/complexes.sage")
load("cobracat-master/braidactions.sage")

###############################
# Set the data locations here #
###############################
PROJECT_NAME='data'
# if running locally, this folder should be created beforehand
DATA_DIR = os.path.join(os.getcwd(), PROJECT_NAME)
DATA_FILE_PAIRS = os.path.join(DATA_DIR, 'candidate_pairs.json')
DATA_FILE_COUNTEREXAMPLES = os.path.join(DATA_DIR, 'counterexamples.json')

############################
# Set the Cartan type here #
############################
#For example, we could input the following for type K4 (the 4-clique).
CARTAN_MATRIX = matrix([[2, -1, -1, -1], [-1, 2, -1, -1], [-1, -1, 2, -1], [-1, -1, -1, 2]])

#########
# Setup #
#########
def zigzag_setup(cartan_matrix):
    """
    Returns dictionaries of the indecomposable projectives and the twists for a given Cartan matrix.

    INPUT:

    - cartan_matrix -- a symmetric Cartan matrix or a CartanType.

    OUTPUT:

    - p -- a dictionary of indecomposable projectives indexed by the index set of the corresponding CartanType
    - s -- a dictionary of spherical twists and inverse spherical twists, indexed by the positives and negatives of the index set of the corresponding CartanType.
    """
    Z = ZigZagAlgebra(cartan_matrix)
    ct = Z.cartan_type
    p = {}
    for i in ct.index_set():
        fpi = ProjectiveZigZagModule(Z, i)
        pi = ProjectiveComplex(Z)
        pi.add_object_at(0, fpi)
        p[i] = pi
    s = {i : (lambda C, i = i: sigma(C, i, Z, minimize=True)) for i in ct.index_set()}
    s |= {-i : (lambda C, i = i: sigma_inverse(C, i, Z, minimize=True)) for i in ct.index_set()}
    return p, s

# Load and return existing data if available.
def initialise_data():
    """
    Returns a list of candidate pairs if available at the global variable DATA_FILE_PAIRS.

    INPUT:

    None

    OUTPUT:

    A list `candidate_pairs`, or ValueError if it does not exist.
    """
    if os.path.exists(DATA_FILE_PAIRS):
        with open(DATA_FILE_PAIRS, 'r') as handle:
            candidate_pairs = json.load(handle)
            print("{} pairs to check".format(len(candidate_pairs)))
            return candidate_pairs
    else:
        raise ValueError("No data")

####################
# Helper functions #
####################

def nb_terms(cplx) :
    count=0
    for i in [cplx.min_index .. cplx.max_index] :
        loclist=cplx.objects[i]
        count=count+len(loclist)
    return count
    return False

# Computes the graded dim of a hom complex
def graded_dim_homs(cplx) :
    count=0
    for i in [cplx.min_index .. cplx.max_index] :
        loclist=cplx.objects[i]
        for x in loclist :
            count+=(-1)^i*q^(x.graded_degree)
    return count


##################
# Main functions #
##################
def dump_counterexamples(cartan_matrix):
    """
    Find pairs of complexes that have more than a single dimensional hom space. Dump these counterexamples in DATA_FILE_COUNTEREXAMPLES.

    INPUT:

    - `cartan_matrix` -- a symmetric Cartan matrix or a CartanType.

    None.

    OUTPUT:

    None. Output is written into DATA_FILE_COUNTEREXAMPLES.
    """
    candidate_pairs = initialise_data()
    p,s = zigzag_setup(cartan_matrix)
    
    counterexamples_found=[]
    pair_count=0
    for pair in candidate_pairs:
        pair_count+=1
        print("Checking a pair {} out of {}. So far we have {} counterexamples.".format(pair_count, len(candidate_pairs),len(counterexamples_found)))
        b1, b2= pair[0][0], pair[1][0]
        base_p1, base_p2 = p[pair[0][1]], p[pair[1][1]]
        # curve1 = composeAll(b1)(base_p1)
        # curve2 = composeAll(b2)(base_p2)
        locstart=time.time()
        curve1=copy(base_p1)
        curve2=copy(base_p2)
        for letter in reversed(b1) :
            curve1=s[letter](curve1)
        for letter in pair[1][0] :
            curve1=s[-letter](curve1)
        print("Created curves in {} seconds, starting computing the hom space".format(time.time()-locstart))
        homsp=curve1.hom(curve2)
        # Sanity check
        print("The Euler characteristics of the hom space is indeed",graded_dim_homs(homsp))
        homsp = homsp.minimize_using_matrix()
        if nb_terms(homsp)!=1 :
            print("Found counterexample",pair)
            counterexamples_found.append(pair)
    print("Found {} counterexamples, here they are:{}".format(len(counterexamples_found),counterexamples_found))

    with open(DATA_FILE_COUNTEREXAMPLES, 'w+') as handle:
        json.dump(counterexamples_found, handle)
