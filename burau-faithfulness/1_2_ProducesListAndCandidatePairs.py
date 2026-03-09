###########
# Imports #
###########
# Standard libraries
import multiprocessing as mp
import time
import numpy as np
import os
import json
from functools import partial

# Local packages
from burau_setup import laurent
from burau_setup import dimvec
from burau_setup.digraph_data import DigraphData

#############################
# Set the Cartan type here. #
#############################
# This is the directed graph of K4 (the 4-clique)
digraph_ct = {1: set([2,3,4]), 2: set([1,3,4]), 3: set([1, 2,4]), 4: set([1,2,3])}
ct = DigraphData(digraph_ct)

###########################
# Set the parameters here #
###########################
# General heuristic computational parameters. Tweak these according to your computational setup and desired shape of search.
MAX_LEN_SPHER=2000
NUM_CPUS=5
PAR_PARAM=1000
Q_BOUND=15
PARALLEL = True

## Enter min and max values for the key (== Burau length) on which to perform the search, for each curve:
MIN_KEY_LEN_1=10
MAX_KEY_LEN_1=20
MIN_KEY_LEN_2=20
MAX_KEY_LEN_2=40

## Enter the desired intersection: 0 if looking for curves with pairing 0, 1 if looking for curves with pairing q^l
INTERSECTION_NUM=1

## Enter the roots that control the class of the curves we want to check
ROOT1='1-0-0-0'
ROOT2='0-1-0-0'


###############################
# Set the data locations here #
###############################
DATA_DIR = os.path.join(os.getcwd(), 'data/')
DATA_FILE = os.path.join(DATA_DIR, 'q-polynomials.json')    
DATA_FILE_PAIRS = os.path.join(DATA_DIR, 'candidate_pairs.json')    

############################
# General helper functions #
############################

def duplicate_rem(seq):
    """
    Function to remove duplicates from entries generated (in parallel) by adding_object_poly_vector.

    INPUT:
    - ``seq``: a sequence of items, each of which has the form [braid, baseP, dim_vec]

    OUTPUT:
    - A list with duplicate dimension vectors removed.
    """
    seen = set()
    result = []
    for item in seq:
        # We look at the dimension vector, which is item[2].
        # Then we convert each entry to a hashable format, and the vector to a tuple.
        # So the whole thing becomes a hashable marker.
        marker = tuple([laurent.to_hashable(d) for d in item[2]])
        if marker in seen:
            continue

        # If new, add marker to seen and item to result.
        seen.add(marker)
        result.append(item)
    return result

# Clean up function:
def clean_dic(dic):
    # Record copy of keys to avoid changing dictionary size during iteration.
    keys_list = list(dic.keys())
    for key in keys_list:
        if len(dic[key])==0 :
            del dic[key]

# Helper function to convert to/from json.
def json_dictionaries_cleanup(d):
    for k in d:
        v = d[k]
        new_v = [[x[0],x[1],dimvec.jsonify(x[2])] for x in v]
        d[k] = new_v
    return d

def initialise_data(ct, create=True):
    """
    Either create new versions of SPHERp and INTERMEDIATEp, or load existing data.

    INPUT:

    - `ct` -- an object of class DigraphData
    - `create` -- a boolean

    OUTPUT:
    
    - `known_data` -- a dictionary of already known data, which may have keys 'SPHERp' and 'INTERMEDIATEp'. 
    - `SPHERp` -- a dictionary of curves indexed by root keys as before.
    - `INTERMEDIATEp` -- a dictionary of curves indexed by root keys.
    """
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as handle:
            known_data = json.load(handle)
    else:
        known_data = {}

    # Read or create SPHERp and INTERMEDIATEp if applicable.
    if 'SPHERp' in known_data:
        SPHERp = json_dictionaries_cleanup(known_data['SPHERp'])
    elif create:
        SPHERp = {}
    else:
        raise ValueError("No SPHERp data")        

    if 'INTERMEDIATEp' in known_data:
        INTERMEDIATEp = json_dictionaries_cleanup(known_data['INTERMEDIATEp'])
    elif create:
        INTERMEDIATEp={}
        for index in range(len(ct.simple_roots)) :
            INTERMEDIATEp[ct.makekey(ct.simple_roots[index],1)]=[[[],index+1,ct.dim_vectors[index+1]]]
    else:
        raise ValueError("No INTERMEDIATEp data")            

    print('Length of SPHERp',sum([len(SPHERp[key]) for key in SPHERp.keys()]))
    print('Length of INTERMEDIATEp',sum([len(INTERMEDIATEp[key]) for key in INTERMEDIATEp.keys()]))
            
    return known_data, SPHERp, INTERMEDIATEp

def braid_relation(b1, b2, ct):
    if abs(b1[0]) != abs(b1[2]) or abs(b2[0]) != abs(b2[2]):
        # Braids not of type aba
        return False
    if abs(b1[0]) != abs(b2[1]) or abs(b1[1]) != abs(b2[0]):
        # Braids not of type aba and bab
        return False
    if not ct.exists_digraph_edge(abs(b1[0]), abs(b1[1])):
        # Braid relation is not expected.
        return False
    if (b1[0] * b1[1] < 0 and b1[1] * b1[2] < 0):
        # Type +-+ or -+-
        return False
    if (b2[0] * b2[1] < 0 and b2[1] * b2[2] < 0):
        # Type +-+ or -+-
        return False
    if (b1[1] == b2[0] and b1[2] == b2[1] and b1[0] * b2[2] > 0):
        # b1 = axy and b2 = xyz, so braiding goes axy -> xyz
        return True
    if (b2[1] == b1[0] and b2[2] == b1[1] and b2[0] * b1[2] > 0):
        # b2 = axy and b1 = xyz, so braiding goes xyz -> axy
        return True
    else:
        return False
    
def duplicate_pair(c1, c2,ct):
    b1, b2 = c1[0], c2[0]
    if len(b1) < 1 or len(b2) < 1:
        # Braids too small.
        return False
    if b1[0] == b2[0]:
        # First letters equal; delete.
        return True
    if len(b1) < 2 or len(b2) < 2:
        # Braids too small
        return False
    if abs(b1[0]) == abs(b2[1]) and abs(b1[1]) == abs(b2[0]) and not(ct.exists_digraph_edge(abs(b1[0]), abs(b1[1]))):
        # First two letters commute to be equal.
        return True
    if len(b1) < 3 or len(b2) < 3:
        # Braids too small
        return False
    if braid_relation(b1[0:3], b2[0:3],ct):
        # There is a braid relation in the first three letters.
        # print(b1, b2)           
        return True
    return False

def key_length(key) :
    ind=-1
    while key[ind]!="-" and ind>-10 :
        ind+=-1
    if ind==-10 :
        raise Exception("No length detected. Check format of the entry.")
    return int(key[ind+1:])

def key_to_root(key) :
    ind=-1
    while key[ind]!="-" and ind>-10 :
        ind+=-1
    if ind==-10 :
        raise Exception("No root detected. Check format of the entry.")
    return key[0:ind]

def burau_vec_to_root(vec,ct) :
    """
    Takes a Burau vector and the digraph and returns the associated root as a vector with integral entries.
    """
    out=np.array([0 for i in range(len(vec))])
    for i in range(len(vec)) :
        out+=sum([(int((-1)**key)*vec[i][key]) for key in vec[i].keys()])*ct.simple_roots[i]
    return out
        
def root_string_to_vec(roo) :
    """
    Convert a root in a string representation to a numpy vector.

    INPUT:

    - `roo` -- a string of coefficients separated by hyphens, for example '1-0-0-0-0'.

    OUTPUT:

    - A numpy array constructed by splitting the string along hyphens and returning the corresponding vector of integers.
    """
    return np.array([int(x) for x in roo.split("-")])

def to_check(pair, ct, inter=INTERSECTION_NUM):
    """
    """
    curve1=pair[0]
    curve2=pair[1]
    if len(ct.pairing(curve1[2],curve2[2]).keys()) == inter:
        return [curve1,curve2]

##########################################################################
# Core functions to generate the actions of braid words on basic curves, #
# and to store them classified by root for later access.                 #
##########################################################################

def adding_object_poly_vector(pair, ct, SPHERp, INTERMEDIATEp):
    """
    Given existing data of a braid applied to one of the generating curves and the old dimension vector, and a new letter of the braid group, apply the new letter and return the new data.

    INPUT:
    
    - ``pair``, a tuple whose first entry is of the form [braid, baseP, old_dim_vec] and second entry is a letter of the braid group.
    - `ct` -- an object of class DigraphData
    - `SPHERp` -- a dictionary of curves indexed by root keys, where a root key is a 6-tuple whose first five entries are the coordinates of the root with respect to the simple roots, and the sixth entry is the length, namely the sum of the absolute values of the coefficients in the Grothendieck group K0.
    - `INTERMEDIATEp`-- a dictionary of curves indexed by root keys.

    OUTPUT:

    A list whose first element is the new braid word, the second element is the generating curve it was applied to, and third element is the new dimension vector.
    """
    # Note that old_dim_vec is a graded dimension vector of Laurent dictionaries.
    [braid,baseP,old_dim_vec]=pair[0]
    x=pair[1]

    if len(braid) >=1 and x + braid[0] == 0:
        "The letter x cancels with the first letter of the braid."
        return None

    # The letter x is supposed to be the next letter in the sequence that we apply.
    # So we apply the corresponding Burau matrix to old_dim_vec, producing new_dim_vec.
    new_dim_vec = ct.burau_fns[x](old_dim_vec)

    # Next we normalize the obtained polynomial.
    new_dim_vec=dimvec.normalize(new_dim_vec)

    # Record the root and number of terms of the new_dim_vec.    
    root=ct.find_ends_vector_mod2(new_dim_vec)
    nb = dimvec.num_terms(new_dim_vec)

    # Make a key using the number of terms.
    lockey=ct.makekey(root,nb)

    # Filter by Q_BOUND
    # We add a bound on the degree to avoid curves the "nb of terms" of which is apparently preserved by a
    # given generator but that actually yield huge curves.
    if dimvec.top_degree(new_dim_vec) > Q_BOUND :
        return None

    # Now check to see if we have already seen this dimension vector in SPHERp.
    # If yes, return None.
    if lockey in SPHERp.keys():
        for y in SPHERp[lockey]:
            if y[2]==new_dim_vec:
                return None

    # Check to see if we have already seen this dimension vector in INTERMEDIATEp.
    # If yes, return None.    
    if lockey in INTERMEDIATEp.keys():
        for y in INTERMEDIATEp[lockey]:
            if y[2]==new_dim_vec :
                return None

    # Otherwise, this is a genuinely new result! Return it.
    return [[x]+braid,baseP,new_dim_vec]


def produce_list(ct):
    """
    Produce a list of braid actions on standard generators, record their classes, and write the data to `DATA_FILE`.

    INPUT:

    - `ct` -- an object of class DigraphData

    OUTPUT:

    None. Writes generated data to `DATA_FILE`.
    """
    known_data, SPHERp, INTERMEDIATEp = initialise_data(ct)

    # Record start time for main loop.
    global_start_time=time.time()

    while sum([len(SPHERp[key]) for key in SPHERp.keys()]) < MAX_LEN_SPHER :
        print('Length of SPHERp at beginning is {}'.format(sum([len(SPHERp[key]) for key in SPHERp.keys()])))
        starttime=time.time()
        for root in ct.roots_mod_2:
            nb=sum(root)
            while nb<30 and (ct.makekey(root,nb) not in INTERMEDIATEp.keys() or len(INTERMEDIATEp[ct.makekey(root,nb)])==0):
                nb=nb+2
            # It might happen (at the end of the algo?) that INTERMEDIATEp is simply empty now for all these entries. In that case we just continue.
            if ct.makekey(root,nb) not in INTERMEDIATEp.keys() :
                continue
            loccap=len(INTERMEDIATEp[ct.makekey(root,nb)])
            NB=nb
            while ct.makekey(root,NB+2) in INTERMEDIATEp.keys() and loccap<PAR_PARAM :
                NB=NB+2
                loccap=loccap+len(INTERMEDIATEp[ct.makekey(root,NB)])
            loccap=min(PAR_PARAM,loccap)
            locchunk=[]
            locsetofobj=[]
            if NB==nb :
                for l in range(loccap):
                    locsetofobj=locsetofobj+[INTERMEDIATEp[ct.makekey(root,nb)][l]]
            if NB>nb and loccap<PAR_PARAM :
                l=nb
                while l<NB :
                    locsetofobj=locsetofobj+[x for x in INTERMEDIATEp[ct.makekey(root,l)]]
                    l=l+2
            if NB>nb and loccap==PAR_PARAM :
                l=nb
                while l<NB :
                    locsetofobj=locsetofobj+[x for x in INTERMEDIATEp[ct.makekey(root,l)]]
                    l=l+2
                l=0
                while len(locsetofobj)<PAR_PARAM :
                    locsetofobj=locsetofobj+[INTERMEDIATEp[ct.makekey(root,NB)][l]]
                    l=l+1
            for y in locsetofobj :
                locchunk=locchunk+[[y,x] for x in ct.restricted_letters]

            # Parallel computing
            if PARALLEL:
                pool=mp.Pool(NUM_CPUS)
                LOC=pool.map(partial(adding_object_poly_vector, ct=ct, SPHERp=SPHERp, INTERMEDIATEp=INTERMEDIATEp), [x for x in locchunk])
                pool.close()
                pool.join()
            else:
                LOC=map(partial(adding_object_poly_vector, ct=ct, SPHERp=SPHERp, INTERMEDIATEp=INTERMEDIATEp), [x for x in locchunk])

            # Removing None entries
            LOC = [x for x in LOC if x is not None]
            LOC=duplicate_rem(LOC)
            # Moving elements to their new place
            dele=loccap
            index=nb
            while dele>0 :
                if len(INTERMEDIATEp[ct.makekey(root,index)])>0 :
                    locdele=min(dele,len(INTERMEDIATEp[ct.makekey(root,index)]))
                    del INTERMEDIATEp[ct.makekey(root,index)][0:locdele]
                    dele=dele-locdele
                else :
                    index=index+2
            for pair in LOC :
                root1=ct.find_ends_vector_mod2(pair[2])
                nb1=dimvec.num_terms(pair[2])
                if ct.makekey(root1,nb1) in INTERMEDIATEp.keys() :
                    INTERMEDIATEp.get(ct.makekey(root1,nb1)).append(pair)
                else :
                    INTERMEDIATEp[ct.makekey(root1,nb1)]=[pair]
            for pair in locsetofobj :
                root1=ct.find_ends_vector_mod2(pair[2])
                nb1=dimvec.num_terms(pair[2])
                if ct.makekey(root1,nb1) in SPHERp.keys() :
                    SPHERp.get(ct.makekey(root1,nb1)).append(pair)
                else :
                    SPHERp[ct.makekey(root1,nb1)]=[pair]
        endtime=time.time()
        print('Length of SPHERp at the end',sum([len(SPHERp[key]) for key in SPHERp.keys()]), 'computation time', endtime-starttime)

    #Record end time of main loop.
    global_end_time=time.time()
    clean_dic(INTERMEDIATEp)
    print("The search took {} seconds.".format(global_end_time-global_start_time))
    print("Length of SPHERp at the end is {}".format(sum([len(SPHERp[key]) for key in SPHERp.keys()])))
    print("Length of INTERMEDIATEp at the end is".format(sum([len(INTERMEDIATEp[key]) for key in INTERMEDIATEp.keys()])))
    # Now dump the new data to our json data storage.
    known_data['SPHERp'] = SPHERp
    known_data['INTERMEDIATEp'] = INTERMEDIATEp

    with open(DATA_FILE, 'w+') as handle:
        json.dump(known_data, handle)

##########################################################################
# Core functions to find and store a list of pairs of big roots with the #
# desired root classes, in order to check their actual and apparent      #
# pairing later on.                                                      #
##########################################################################

def find_big_pairs(ct, root_1, root_2):
    """
    Return a list of "big pairs", such that the first entry has class `root_1` and the second entry has class `root_2`.

    Moreover, the length of each element of the pair is governed by `MIN_KEY_LEN_1`, `MAX_KEY_LEN_1`, `MIN_KEY_LEN_2`, and
    `MAX_KEY_LEN_2` respectively.
    The pairs are taken from elements of `SPHERp` and `INTERMEDIATEp`.

    INPUT:

    - `ct` -- an object of class DigraphData
    - `root_1`
    - `root_2`

    OUTPUT:

    - `all_big_pairs` -- a list of pairs of big roots
    """

    # Initialise data.
    known_data, SPHERp, INTERMEDIATEp = initialise_data(ct, create=False)

    # Preparing the lists of curves to check
    #Curves1=[]
    big_curves_1=[]
    #Curves2=[]
    big_curves_2=[]
    
    ## roo1 and roo2 are given as strings. If we want to filter out curves that exactly decategorify to these roots (and not just mod 2), we should turn them into np.array

    root_ref_1=root_string_to_vec(root_1)
    root_ref_2=root_string_to_vec(root_2)

    num_curves_1=0
    num_curves_2=0
    num_filtered_out=0
    
    for key in list(SPHERp.keys()) :
        if key_to_root(key)==root_1 :
            num_curves_1+=len(SPHERp[key])
            if key_length(key)>=MIN_KEY_LEN_1 and key_length(key)<=MAX_KEY_LEN_1 :
                for curve in SPHERp[key] :
                    num_filtered_out+=1
                    if np.all(burau_vec_to_root(curve[2],ct)==root_ref_1) or np.all(burau_vec_to_root(curve[2],ct)==-root_ref_1) :
                        num_filtered_out+=-1
                        big_curves_1.append(curve)
        if key_to_root(key)==root_2 :
            num_curves_2+=len(SPHERp[key])
            if key_length(key)>=MIN_KEY_LEN_2 and key_length(key)<=MAX_KEY_LEN_2 :
                for curve in SPHERp[key] :
                    # num_filtered_out+=1
                    # if np.all(burau_vec_to_root(curve[2])==root_ref_2) or np.all(burau_vec_to_root(curve[2])==-root_ref_2) :
                    #     num_filtered_out+=-1
                    #     big_curves_2.append(curve)
                    big_curves_2.append(curve)

    for key in list(INTERMEDIATEp.keys()) :
        if key_to_root(key)==root_1 :
            num_curves_1+=len(INTERMEDIATEp[key])
            if key_length(key)>=MIN_KEY_LEN_1 and key_length(key)<=MAX_KEY_LEN_1 :
                for curve in INTERMEDIATEp[key] :
                    num_filtered_out+=1
                    if np.all(burau_vec_to_root(curve[2],ct)==root_ref_1) or np.all(burau_vec_to_root(curve[2],ct)==-root_ref_1) :
                        num_filtered_out+=-1
                        big_curves_1.append(curve)
        if key_to_root(key)==root_2 :
            num_curves_2+=len(INTERMEDIATEp[key])
            if key_length(key)>=MIN_KEY_LEN_2  and key_length(key)<=MAX_KEY_LEN_2 :
                for curve in INTERMEDIATEp[key] :
                    # num_filtered_out+=1
                    # if np.all(burau_vec_to_root(curve[2])==root_ref_2) or np.all(burau_vec_to_root(curve[2])==-root_ref_2) :
                    #     num_filtered_out+=-1
                    #     big_curves_2.append(curve)
                    big_curves_2.append(curve)

    print("All curves:",num_curves_1,num_curves_2, "Big curves", len(big_curves_1),len(big_curves_2),"Filtered ",num_filtered_out)

    all_big_pairs = ([c1, c2] for c1 in big_curves_1 for c2 in big_curves_2 if not duplicate_pair(c1, c2,ct))

    print("There are {} pairs remaining to check.".format(len(big_curves_1)*len(big_curves_2)))
    return(all_big_pairs)

def check_next_chunk(ct,all_big_pairs):
    if os.path.exists(DATA_FILE_PAIRS):
        with open(DATA_FILE_PAIRS, 'r') as handle:
            candidate_pairs = json.load(handle)
    else :
        candidate_pairs=[]

    print("Loaded old candidate pairs: there are {} pairs.".format(len(candidate_pairs)))
    local_start_time=time.time()
    
    count = 0
    chunk = []
    while count < 1000000:
        try:
            chunk.append(next(all_big_pairs))
            count = count + 1
        except StopIteration:
            break
    if chunk == []:
        print("Exhausted all_big_pairs.")
        return False
    else:
        print("Now checking new chunk of size {}".format(len(chunk)))
        
    pool = mp.Pool(NUM_CPUS)
    new_candidate_pairs = pool.map(partial(to_check, ct=ct), chunk, chunksize=2000)
    new_candidate_pairs = [x for x in new_candidate_pairs if x is not None]
    print("Found {} new candidate pairs in {} seconds using parallel version".format(len(new_candidate_pairs),time.time()-local_start_time))
    for p in new_candidate_pairs:
        if p not in candidate_pairs:
            candidate_pairs.append(p)

    with open(DATA_FILE_PAIRS, 'w+') as handle:
        json.dump(candidate_pairs, handle)


def find_candidate_pairs(ct):
    local_start_time=time.time()
    
    all_big_pairs=find_big_pairs(ct,ROOT1,ROOT2)

    candidate_pairs=[]
    cond=True
    gencounter=0
    while cond==True :
        gencounter+=1
        print("Starting step number",gencounter)
        outcome=check_next_chunk(ct,all_big_pairs)
        if outcome==False :
            cond=False
            print("Stopping")

###############################
# Run the main functions here #
###############################
# Produce the list. Note that the intial data is created if it does not exist.
produce_list(ct)

## Search for candidate pairs. Note that the initial data is simply loaded, not created.
find_candidate_pairs(ct)
