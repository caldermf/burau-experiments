print('Beginning of the file')

# Imports
import random
import multiprocessing as mp
import time
import numpy as np
import os
import json
# Import local packages
import laurent
import setup_d4 as ct




# Parameters
# CAP_1 and CAP_2 are caps on the bucket sizes. We can specify a value for the first few steps (presumably larger?) then another one for the next steps.
# Because we use these values on each worker, the actual caps for bucket sizes are actually these values multiplied by the number of CPU's.
CAP_1=500 
CAP_2=500
# TOTAL_CAP_1 and TOTAL_CAP_2 are caps on the total number of curves we keep at a given iteration. We can specify two values, for the first steps and for the remaining ones.
TOTAL_CAP_1=50000
TOTAL_CAP_2=50000
# FIRST_STEPS specifies when to switch from CAP_1 to CAP_2
FIRST_STEPS=12 # number of steps where we explore largely
# PP is the value to specify to work over Z/pZ. PP=0 means working over Z
PP=7
# Maximum Garside lengths of braids to consider
MAX_G_LENGTH=1000
# Number of CPU's to be used for parallel computing.
NB_CPU=8

print('Starting D4 with p=',PP,"caps",CAP_1,CAP_2,"totalcaps",TOTAL_CAP_1,TOTAL_CAP_2,"FIRST_STEPS",FIRST_STEPS,"NB_CPU",NB_CPU)


## The following set of parameters finds a counterexample for PP=7 in 160s
## CAP_1=500, CAP_2=500, TOTAL_CAP_1=50000, TOTAL_CAP_2=50000, FIRST_STEPS=12, PP=7, MAX_G_LENGTH=1000, NB_CPU=8



globalstart=time.time()





    


def ComputeOBurauVector(br,baseP) :
    """
    Given a braid and the index of a base vector, computes the image of the base vector under the braid, with the orientation grading conventions.

    INPUT:
    - br a list of braid generators (positive or negative integers corresponding to positive or negative powers of the Artin generators)
    - baseP the index of a basis vector, interpreted as a key of dim_vectors from setup_d5

    OUTPUT:
    - the image of baseP under br, as a dimension vector.
    """
    
    new_dim_vec=ct.dim_vectors[baseP]
    for u in reversed(br) :
        new_dim_vec = ct.oburau_fns[u](new_dim_vec)
    return new_dim_vec


# Dual generators with 1 --> 2 --> 3
#                              --> 4
# Working with negative powers, so that the degree is positive
Duals=[[-1],[-2],[-3],[-4],[1,-2,-1],[2,-3,-2],[2,-4,-2],[1,2,-3,-2,-1],[1,2,-4,-2,-1],[-3,-4,-2,4,3],
       [-4,-3,1,-2,-1,3,4],[2,-4,-3,1,-2,-1,3,4,-2]]

print("We have {} atoms, PP is {}".format(len(Duals),PP))

def ComputeOBurauDeg(braid) :
    """
    Computes the top degree of the matrix entries for the braid.
    """
    return max([ct.topdeg_vector(ComputeOBurauVector(braid,baseP)) for baseP in [1,2,3,4]])

def EqualBraids(braid1,braid2) :
    """
    Checks if two braids have the same oriented Burau matrix.
    """
    for baseP in [1,2,3,4] :
        if ComputeOBurauVector(braid1,baseP)!=ComputeOBurauVector(braid2,baseP) :
            return False
    return True

GarsideGens=[[]]+Duals

# Adding linear products until we find no more: this produces the Garside simples.
number=1
while number!=0 :
    ToAdd=[]
    for x in GarsideGens :
        for y in GarsideGens :
            if ComputeOBurauDeg(x+y) <=1 : # Here because we're working with small braids, we use the Burau matrices to compare braids.
                test=True
                for u in ToAdd :
                    if EqualBraids(u,x+y)==True :
                        #Here we're trying to store the lowest length form for a given Garside generator
                        if len(x+y)<len(u) :
                            ToAdd.remove(u)
                            ToAdd.append(x+y)
                        test=False
                for u in GarsideGens :
                    if EqualBraids(u,x+y)==True :
                        if len(x+y)<len(u) :
                            GarsideGens.remove(u)
                            GarsideGens.append(x+y)
                        test=False
                if test==True :
                    ToAdd.append(x+y)
    GarsideGens=GarsideGens+ToAdd
    print("Added {} generators".format(len(ToAdd)))
    number=len(ToAdd)



    

# Given the left-most letter x_n of a Garside NF, we want to know what letter x_n+1 can be applied to the left.
# Such a letter is so that all of its descents is a right descent of x_n
# First, let's list left and right descents of all letters. We take as atoms the generators from Duals.
LeftDescents={} # Keys will be string versions of the braids.
RightDescents={}

# Given a braid, we need to know which representative is in GarsideGens :
def findRepresentative(braid) :
    """
    Finds the key under which the braid appears in our list of simples.
    """
    for x in GarsideGens :
        if EqualBraids(x,braid) :
            return(x)
    raise Exception('No match found')



### Produces automata for right and left descents.
for descent in Duals :
    for gen in GarsideGens :
        if ComputeOBurauDeg(gen+descent)<=1 :
            y=findRepresentative(gen+descent)
            if str(y) in RightDescents.keys() :
                RightDescents[str(y)].append(descent)
            else :
                RightDescents[str(y)]=[descent]
        if ComputeOBurauDeg(descent+gen)<=1 :
            y=findRepresentative(descent+gen)
            if str(y) in LeftDescents.keys() :
                LeftDescents[str(y)].append(descent)
            else :
                LeftDescents[str(y)]=[descent]



### Computes the Garside automaton
print("Started computing the automaton")
GarsideAutomaton={}
for y in GarsideGens :
    GarsideAutomaton[str(y)]=[]
    if y!=[] and y!=findRepresentative([-4,-3,-2,-1]) :
        for x in GarsideGens :
            if x!=[] and y !=findRepresentative([-4,-3,-2,-1]) :
                test=True
                for descent in RightDescents[str(x)] :
                    if descent!=[] and descent!=findRepresentative([-4,-3,-2,-1]) :
                        if ComputeOBurauDeg(descent+y) <=1 :
                            test=False
                if test :
                    GarsideAutomaton[str(y)].append(x)
print("Done with automaton computation")

def spreadofvectors(family) :
    """
    Given a family of Burau vectors, returns the global spread.
    """
    TOP=[ct.topdeg_vector(vect) for vect in family]
    BOT=[ct.botdeg_vector(vect) for vect in family]
    return max(TOP)-min(BOT)





def makeFp(pol,p) :
    """
    Takes a dimension poly and returns the same one mod Fp.

    INPUT:
    - a q-polynomial pol as a dictionary
    - an integer p
    
    OUTPUT:
    - the q-polynomial with coefficients reduced modulo p.
    """
    if p==0 :
        return pol
    out={}
    for key in pol.keys() :
        coeff=pol[key]
        newcoeff=coeff%p
        if newcoeff!=0 :
            out[key]=newcoeff
    return out

def makeFpVec(vec,p) :
    """
    Takes a dimension vector and returns the same one mod Fp, by applying makeFP to each of the entries.
    """
    return [makeFp(pol,p) for pol in vec]



### Main function
def iterate(entrylist) :
    """
    Given an entry in a bucket, applies all admissible simples, looks if we've found a non-trivial curve with spread 0, and returns the new curves together with the number of drops in spread (for information).
    """
    out={}
    if len(entrylist)>0 :
        prevdeg=ct.topdeg_vector(entrylist[0][1])-ct.botdeg_vector(entrylist[0][1]) # They all have the same degree, so doing this only once.
    locdrops=0
    for prevelt in entrylist :
        for x in GarsideAutomaton[str(prevelt[0][0])] :
            locburau=prevelt[1].copy()
            for u in reversed(x) :
                locburau=makeFpVec(ct.oburau_fns[u](locburau),PP)
            locspread=ct.topdeg_vector(locburau)-ct.botdeg_vector(locburau)
            if cur!=2 or (cur==2 and locspread!=0) : # This is to rule out stupid lifts
                if locspread==0 :
                    stop=True
                    print("Found one for p=",PP,[x]+prevelt[0])
                if locspread<prevdeg :
                    locdrops+=1
                if locspread<=MAX_G_LENGTH-cur+1 :
                    if locspread in out.keys() :
                        if len(list(out[locspread]))<CAP : 
                            out[locspread].append([[x]+prevelt[0],locburau])
                        else :
                            position=random.choice(range(CAP+1))
                            if position<CAP :
                                out[locspread][position]=[[x]+prevelt[0],locburau]
                    else :
                        out[locspread]=[[[x]+prevelt[0],locburau]]
    return([out,locdrops])






### Initializing with all Garside generators
### We start filling the buckets with all images of v1 under simples that respect the following property: bv1 has spread 1.
B={}
cur=1
B[cur]={}
for x in GarsideGens :
    if x!=[] and x!= findRepresentative([-4,-3,-2,-1]) :
        test=True
        for descent in RightDescents[str(x)] :
            if ct.poly_normalize_vector(ComputeOBurauVector(descent,1))==ct.dim_vectors[1] :
                test=False
            if ct.topdeg_vector(ComputeOBurauVector(descent,1))-ct.botdeg_vector(ComputeOBurauVector(descent,1))!=1 :
                test=False
        if test :
            locburau=makeFpVec(ComputeOBurauVector(x,1),PP)
            locspread=ct.topdeg_vector(locburau)-ct.botdeg_vector(locburau)
            if locspread==1 : # we add this condition to maintain minimality
                if locspread in B[cur].keys() :
                    B[cur][locspread].append([[x],locburau])
                else :
                    B[cur][locspread]=[[[x],locburau]]





### We propagate the buckets by applying to all entries from the previous steps all simples allowed in the Garside automaton.
stop=False
while cur<MAX_G_LENGTH and stop==False :
    starttime=time.time()
    cur+=1 
    print("Starting step",cur)
    B[cur]={} #Initializing the bucket dictionary
    nbdrops=0 # Initializing counter of spread drops
    if cur<FIRST_STEPS :
        TOTAL_CAP=TOTAL_CAP_1
        CAP=CAP_1
    else :
        TOTAL_CAP=TOTAL_CAP_2
        CAP=CAP_2
    # We'll only analyze a number of curves less than TOTAL_CAP: here we find which buckets from the previous steps we should consider, keeping only those with lower spread.
    stop_key=min(list(B[cur-1].keys()))
    counter=len(list(B[cur-1][stop_key]))
    while counter<TOTAL_CAP and stop_key+1 in B[cur-1].keys() :
        stop_key+=1
        counter+=len(list(B[cur-1][stop_key]))
    keylist=[key for key in B[cur-1].keys() if key<=stop_key]
    keylist.sort()
    ### Starting filling in the new buckets
    for prevdeg in keylist :
        print("Analyzing {} braids in spread {}".format(len(B[cur-1][prevdeg]),prevdeg))
        ### Preparing chunks for parallel computing
        listoflists=[[B[cur-1][prevdeg][index] for index in range(len(B[cur-1][prevdeg])) if index%NB_CPU==modcpu] for modcpu in range(NB_CPU)]
        ### Parallel computing
        pool=mp.Pool(NB_CPU)
        results=pool.map(iterate, listoflists)
        pool.close()
        pool.join()
        ### Assembling all dictionaries:
        for locentry in results :
            locdic=locentry[0]
            nbdrops+=locentry[1]
            for lockey in locdic.keys() :
                if lockey not in B[cur].keys() :
                    B[cur][lockey]=locdic[lockey].copy()
                    # To release memory
                    locdic[lockey]=[]
                else :
                    if len(B[cur][lockey])<CAP*(NB_CPU-1) : 
                        B[cur][lockey]+=locdic[lockey]
                        # To release memory
                        locdic[lockey]=[]
                    else :
                        for elt in locdic[lockey] :
                            if len(B[cur][lockey])<CAP*NB_CPU :
                                B[cur][lockey].append(elt)
                            else :
                                position=random.choice(range(NB_CPU*CAP+1))
                                if position<NB_CPU*CAP :
                                    B[cur][lockey][position]=elt
                        # To release memory
                        locdic[lockey]=[]
    ### Stop condition (turning stop to True in the iterate function doesn't seem to work)
    if 0 in B[cur].keys() :
        stop=True
    # Freeing up memory
    B[cur-1]={}
    endtime=time.time()
    print("Finished with step {}, minimal spread is {}, max spread is {}, got {} drops".format(cur,min(list(B[cur].keys())),max(list(B[cur].keys())),nbdrops))
    print("Total number of curves", sum([len(B[cur][key])  for key in B[cur].keys()]))
    print("Took {} seconds".format(endtime-starttime))

print("Altogether, runtime=",time.time()-globalstart)

