import numpy as np
from sympy.logic import SOPform, boolalg
from sympy import symbols, srepr, Not
from tqdm import tqdm
import matplotlib.pyplot as plt
from BoolFuncModule.BoolFunc import BoolFunc
            
def main():
    
    sampleSize = 1000
    parameterType = 3

    frequencyDict = {}
    for i in tqdm(range(sampleSize)):
        function = BoolFunc(n=3, parameterType=parameterType)
        isInDict = False
        SOP = function.get_SOP()
        if SOP is True or SOP is False or not isinstance(SOP, boolalg.Or):      # accounts for SOPs with just one literal for which equals() function seems to fail.
            for key in frequencyDict:
                if function.get_SOP() == key:
                    frequencyDict[key][0] += 1
                    isInDict = True
                    break
        else:
            for key in frequencyDict:
                if function.get_SOP().equals(key):
                    frequencyDict[key][0] += 1
                    isInDict = True
                    break
        if isInDict is False:
            frequencyDict[SOP] = [1, function.get_D()]
    
    
    P = [val[0]/sampleSize for key, val in frequencyDict.items()]
    D = [val[1] for key, val in frequencyDict.items()]

    x = np.linspace(0, max(D), 100)
    y = 0.15*2**(-x)

    plt.scatter(D, P)
    plt.plot(x,y, linestyle='--', label='$P\propto2^{-D}$')
    plt.yscale('log', base=2)
    plt.title(f'{sampleSize} samples for type {parameterType} NN')
    plt.xlabel('D')
    plt.ylabel('P')
    plt.legend()
    plt.savefig(f'type{parameterType}/P_vs_D_samples{sampleSize}.png')
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    