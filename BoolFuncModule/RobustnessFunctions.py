import numpy as np
from . import NNfunctions as NNf

def binary(n, length=1):
    # Returns binary string for input n
    if n == 0:
        return length*'0'
    nums = []
    while n:
        n, r = divmod(n, 2)
        nums.append(str(r))
    if length > len(nums):
        for i in range(length-len(nums)):
            nums.append('0')
    return ''.join(reversed(nums))

def ternary(n, length=1):
     # Returns ternary string for input n
    if n == 0:
        return length*'0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    if length > len(nums):
        for i in range(length-len(nums)):
            nums.append('0')
    return ''.join(reversed(nums))

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

def truth_table(parameters):
    W1 = parameters[0]
    n_input = W1.shape[1]

    truthTable = ''
    for i in range(2**n_input):
        i_binaryRep = binary(i, length=n_input)
        input = np.zeros((n_input,1), dtype=int)
        for j, char in enumerate(i_binaryRep):
            input[j,0] = int(char)
        output = NNf.forward_pass(parameters, input)
        truthTable += str(output.item())
    
    return truthTable

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

def mutate_W2(W2, b2, position):
    # Returns all possible mutations of b2 matrix for a point mutation
    if not isinstance(position, int):
        raise TypeError('position must be an integer.')
    if ((1 in W2) and b2 == 1) or ((-1 in W2) and b2 == 0):
        raise Exception('Incompatible W2 and b2 matrices')
    currentVal = W2[0,position]
    if b2 == 0:
        for newVal in [0, 1]:
            if newVal != currentVal:
                mutated_W2 = np.copy(W2)
                mutated_W2[0,position] = newVal
                return mutated_W2
    elif b2 == 1:
        for newVal in [-1, 0]:
            if newVal != currentVal:
                mutated_W2 = np.copy(W2)
                mutated_W2[0,position] = newVal
                return mutated_W2
    else:
        raise Exception('Invalid value for b2. Must be 0 or 1.')
    
def robustness(parameters):
    W1, b1, W2, b2 = parameters
    functionTruthTable = truth_table(parameters)
    robustCount = 0
    notRobustCount = 0
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_mutations = mutate_W1(W1, (i,j))
            for new_W1 in W1_mutations:
                newParameters = (new_W1, b1, W2, b2)
                if functionTruthTable == truth_table(newParameters):
                    robustCount += 1
                else:
                    notRobustCount += 1
    for i in range(W2.shape[1]):
        new_W2 = mutate_W2(W2, b2, i)
        newParameters = (W1, b1, new_W2, b2)
        if functionTruthTable == truth_table(newParameters):
            robustCount += 1
        else:
            notRobustCount += 1
    robustness = robustCount/(robustCount + notRobustCount)
    return robustness




#def estimate_robustness(parameters)