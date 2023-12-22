import numpy as np
from . import NNfunctions as NNf

def is_hamming_pair(string1, string2):
    if len(string1) != len(string2):
        raise Exception('Input string sizes do not match')
    if string1 == string2:
        return False
    count_diffs = 0
    for a, b in zip(string1, string2):
        if a!=b:
            if count_diffs:
                return False
            else:
                count_diffs += 1
    return True

def mutate_W1(W1, position):
    # Returns all possible mutations of W1 matrix for a point mutation
    if not isinstance(position, tuple):
        raise TypeError('position must be a tuple.')
    n, m = position
    currentVal = W1[n,m]
    mutated_W1s = []
    for newVal in [-1, 0, 1]:
        if newVal != currentVal:
            new_W1 = np.copy(W1)
            new_W1[n,m] = newVal
            mutated_W1s.append(new_W1)
    return mutated_W1s

def mutate_b1(b1, position):
    currentVal = b1[position,0]
    mutated_b1s = []
    for newVal in [-1, 0, 1]:
        if newVal != currentVal:
            new_b1 = np.copy(b1)
            new_b1[position, 0] = newVal
            mutated_b1s.append(new_b1)
    return mutated_b1s

def mutate_W2(W2, b2, position, base3=False):
    # Returns all possible mutations of b2 matrix for a point mutation
    if not isinstance(position, int):
        raise TypeError('position must be an integer.')
    #if ((1 in W2) and b2 == 1) or ((-1 in W2) and b2 == 0):
    #    raise Exception('Incompatible W2 and b2 matrices')
    currentVal = W2[0,position]
    if base3 is False:
        if b2 == 0:
            mutated_W2s = []
            for newVal in [0, 1]:
                if newVal != currentVal:
                    new_W2 = np.copy(W2)
                    new_W2[0, position] = newVal
                    mutated_W2s.append(new_W2)
            return mutated_W2s
        elif b2 == 1:
            mutated_W2s = []
            for newVal in [-1, 0]:
                if newVal != currentVal:
                    new_W2 = np.copy(W2)
                    new_W2[0, position] = newVal
                    mutated_W2s.append(new_W2)
            return mutated_W2s
        else:
            raise Exception('Invalid value for b2. Must be 0 or 1.')
    if base3 is True:
        mutated_W2s = []
        for newVal in [-1, 0, 1]:
            if newVal != currentVal:
                new_W2 = np.copy(W2)
                new_W2[0, position] = newVal
                mutated_W2s.append(new_W2)
        return mutated_W2s
    
def robustness(parameters, parameterType):
    W1, b1, W2, b2 = parameters
    functionTruthTable = NNf.truth_table(parameters)
    robustCount = 0
    notRobustCount = 0
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_mutations = mutate_W1(W1, (i,j))
            for new_W1 in W1_mutations:
                newParameters = (new_W1, b1, W2, b2)
                if functionTruthTable == NNf.truth_table(newParameters):
                    robustCount += 1
                else:
                    notRobustCount += 1
    if parameterType == 4:
        for i in range(b1.shape[0]):
            b1_mutations = mutate_b1(b1, i)
            for new_b1 in b1_mutations:
                newParameters = (W1, new_b1, W2, b2)
                if functionTruthTable == NNf.truth_table(newParameters):
                    robustCount += 1
                else:
                    notRobustCount += 1
    isBase3 = True if parameterType == 1 or parameterType == 4 else False
    for i in range(W2.shape[1]):
        W2_mutations = mutate_W2(W2, b2, i, base3=isBase3)
        for new_W2 in W2_mutations:
            newParameters = (W1, b1, new_W2, b2)
            if functionTruthTable == NNf.truth_table(newParameters):
                robustCount += 1
            else:
                notRobustCount += 1
    robustness = robustCount/(robustCount + notRobustCount)
    return robustness