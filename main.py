import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sympy.logic import SOPform, boolalg
from sympy import symbols, srepr, Not
from BoolFunc import BoolFunc
            
def main():
    
    sample_size = 1000
    freq_dict = {}
    for i in tqdm(range(sample_size)):
        func = BoolFunc(n=3, param_type=3)
        count = 0
        SOP = func.get_SOP()
        if SOP == True or SOP == False or not isinstance(SOP, boolalg.Or):      # accounts for SOPs with just one literal for which equals() function seems to fail.
            for key in freq_dict:
                if func.get_SOP() == key:
                    freq_dict[key][0] += 1
                    count += 1
                    break
        else:
            for key in freq_dict:
                if func.get_SOP().equals(key):
                    freq_dict[key][0] += 1
                    count += 1
                    break
        if count == 0:
            freq_dict[SOP] = [1, func.get_D()]
    
    
    P = []
    D = []
    for key in freq_dict:
        P.append(freq_dict[key][0]/sample_size)
        D.append(freq_dict[key][1])
        
    x = np.linspace(1,12, 100)
    y = 2**(-x)
    
    plt.scatter(D, P)
    plt.plot(x,y, linestyle='--', label='$P=2^{-D}$')
    plt.yscale('log')
    plt.xlabel('D')
    plt.ylabel('P')
    plt.legend()
    plt.savefig('P_vs_D.png')
    
    
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    