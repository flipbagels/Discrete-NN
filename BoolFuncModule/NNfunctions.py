import numpy as np

def relu(x):
    return x * (x > 0)

def sign(x):
    return 1 * (x > 0)

def forward_pass(parameters, x):
    if not isinstance(parameters, tuple):
        raise TypeError('parameters must be tuple.')
    W1, b1, W2, b2 = parameters
    x = W1@x + b1
    x = relu(x)
    x = W2@x + b2
    x = sign(x)
    return x

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

def biternary(n, length_binary, length_ternary):
    # Returns a string to represent an input n in the form '{binary}'+'{ternary}' where length_binary (length_ternary) indicate the number of binary (ternary) digits
    length = length_binary + length_ternary
    if n==0:
        return length*'0'
    nums = []
    for _ in range(length_ternary):
        n, r = divmod(n, 3)
        nums.append(str(r))
    while n:
        n, r = divmod(n, 2)
        nums.append(str(r))
    if length > len(nums):
        for i in range(length-len(nums)):
            nums.append('0')
    return ''.join(reversed(nums))

def truth_table(parameters):
    W1 = parameters[0]
    n_input = W1.shape[1]

    truthTable = ''
    for i in range(2**n_input):
        i_binaryRep = binary(i, length=n_input)
        input = np.zeros((n_input,1), dtype=int)
        for j, char in enumerate(i_binaryRep):
            input[j,0] = int(char)
        output = forward_pass(parameters, input)
        truthTable += str(output.item())
    
    return truthTable