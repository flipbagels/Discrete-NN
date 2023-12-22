import numpy as np
from sympy.logic import SOPform, boolalg
from sympy import symbols, srepr, Not
import re
from . import NNfunctions as NNf


def literals_clauses_count(functionSOP, n_input):
    force = False if n_input < 8 else True
    conjugateFunctionSOP = boolalg.to_dnf(Not(functionSOP), simplify=True, force=force)
    sfunctionSOP = srepr(functionSOP)
    sconjugateFunctionSOP = srepr(conjugateFunctionSOP)
    
    # Find number of literals in both SOP and Not(SOP)
    literalsCount = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('Symbol'), sfunctionSOP))
    conjugateLiteralsCount = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('Symbol'), sconjugateFunctionSOP))

    # Choose the expression that minimises the number of literals
    if literalsCount < conjugateLiteralsCount:
        if functionSOP == True or functionSOP == False:
            return 0, 0

        elif not isinstance(functionSOP, boolalg.Or):
            clausesCount = 1
            return literalsCount, clausesCount

        else:
            clausesCount = len(functionSOP.args)
            return literalsCount, clausesCount
    else:
        if conjugateFunctionSOP == True or conjugateFunctionSOP == False:
            return 0, 0

        elif not isinstance(conjugateFunctionSOP, boolalg.Or):
            conjugateClausesCount = 1
            return conjugateLiteralsCount, conjugateClausesCount

        else:
            conjugateClausesCount = len(conjugateFunctionSOP.args)
            return conjugateLiteralsCount, conjugateClausesCount
    

def function_SOP(parameters, n_input):
    W1, b1, W2, b2 = parameters
    truthTable = NNf.truth_table(parameters)
    symbolsList = [symbols(f'{i}') for i in range(n_input)]
    minterms = []
    for i, val in enumerate(truthTable):
        if val == '1':
            minterms.append(list(map(int, NNf.binary(i, length=n_input))))

    if b2 == 0:
        return SOPform(symbolsList, minterms)
    elif b2 == 1:
        force = False if n_input < 8 else True
        return boolalg.to_dnf(Not(SOPform(symbolsList, minterms)), simplify=True, force=force)
    else:
        raise Exception('b2 has no valid value, must be 0 or 1.')


# def function_SOP(parameters, n_input):
#     W1, b1, W2, b2 = parameters
#     symbolsList = [symbols(f'{i}') for i in range(n_input)]

#     # Standard SymPy notation
#     minterms = []
#     for i in range(2**(n_input-1)):
#         if ((-1 in W1[i,:]) or (1 in W1[i,:])) and W2[0,i] != 0:
#             minterms.append({})
#             for j in range(n_input):
#                 if W1[i,j] == 1:
#                     minterms[-1][symbolsList[j]] = 1
#                 elif W1[i,j] == -1:
#                     minterms[-1][symbolsList[j]] = 0
#                 else:
#                     pass

#     if b2 == 0:
#         return SOPform(symbolsList, minterms)
#     elif b2 == 1:
#         force = False if n_input < 8 else True
#         return boolalg.to_dnf(Not(SOPform(symbolsList, minterms)), simplify=True, force=force)
#     else:
#         raise Exception('b2 has no valid value, must be 0 or 1.')