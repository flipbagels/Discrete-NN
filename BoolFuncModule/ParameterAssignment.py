import numpy as np

def uniform_matrix(size, positive=False):
    # Function that returns a matrix with random integers between -1 and 1
    if not isinstance(size, tuple):
        raise TypeError('size must be tuple.')
    if not positive:
        return np.random.randint(-1, 2, size=size)
    elif positive:
        return np.random.randint(0, 2, size=size)
    else:
        raise Exception('Invalid value for positive key word argument. Argument takes either True or False.')


def assign_b1_from_W1(W1):
    n_input = W1.shape[1]
    b1 = np.zeros((2**(n_input-1),1), dtype=int)
    for i in range(2**(n_input-1)):
        if (-1 in W1[i,:]) or (1 in W1[i,:]):
            onesInW1Row = 0
            for j in W1[i,:]:
                if j == 1:
                    onesInW1Row += 1
            b1[i,0] = 1 - onesInW1Row
        else:
            b1[i,0] = 0
    return b1

def assign_W2_from_W1(W1):
    # Assigns 1 for each row in W1 that contains any non zero elements.
    n_input = W1.shape[1]
    W2 = np.zeros((1,2**(n_input-1)), dtype=int)
    for i in range(2**(n_input-1)):
        W2[0,i] = (-1 in W1[i,:]) or (1 in W1[i,:])
    return W2

def noo_random_matrix(size):
    # Number of ones is uniformly determined and then distributed randomly into matrix.
    # Elements take either 1 or 0 based on this assignment
    if not isinstance(size, tuple):
        raise TypeError('size must be tuple.')
    dim = size[0]*size[1]
    numberOfOnes = np.random.randint(0, dim+1)
    onesArray = np.array([1]*numberOfOnes+[0]*(dim-numberOfOnes))
    np.random.shuffle(onesArray)
    W2 = onesArray.reshape(size[0], size[1])
    return W2

def zero_or_one():
    # returns 0 or 1 with 50% probability
    if np.random.rand() > 0.5:
        return 1
    else:
        return 0