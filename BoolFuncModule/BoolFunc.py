import numpy as np
from sympy.logic import SOPform, boolalg
from sympy import symbols, srepr, Not
import re
from . import SOPfunctions as SOPf  # Dot notation as Python imports 'as' commands as relative imports
from . import ParameterAssignment as PA
from . import NNfunctions as NNf
from . import RobustnessFunctions as RF

class BoolFunc():
    def __init__(self, parameters=None, n=3, parameterType=0):
        if parameters is None:
            self.parameters = ()
        if not isinstance(n, int):
            raise TypeError('n must be an int.')
        if not isinstance(parameterType, int) or n >= 4:
            raise Exception('parameterType must be an integer between 0 and 3')
        self.n = n   # input size
        self.parameterType = parameterType
        self.assign_parameters(parameters)
    
    def get_parameters(self):
        return self.parameters
    
    def get_param_type(self):
        return self.parameterType
    
    def change_parameter_type(self, x):
        if x >= 4:
            raise Exception('param_type must be a number between 0 and 3.')
        self.parameterType = x
        self.assign_parameters()
        print(f'Neural network changed to type {x}')
    
    def forward(self, x):
        return NNf.forward_pass(self.parameters, x)
    
    def update_D(self):
        if self.parameterType == 0:
            raise Exception('Not supported yet.')
        
        if self.parameterType == 1:
            parameters = self.get_parameters()
            functionSOP = SOPf.function_SOP(parameters, self.n)
            literalsCount, clausesCount = SOPf.literals_clauses_count(functionSOP, self.n)
            self.D = literalsCount
        
        if self.parameterType == 2 or self.parameterType == 3:    
            parameters = self.get_parameters()
            functionSOP = SOPf.function_SOP(parameters, self.n)
            literalsCount, clausesCount = SOPf.literals_clauses_count(functionSOP, self.n)
            self.D = literalsCount + clausesCount
            
        
    def get_D(self):
        return self.D
    
    def get_SOP(self):
        if self.parameterType == 0:
            raise Exception('Not supported yet.')
        
        if self.parameterType == 1 or self.parameterType == 2 or self.parameterType == 3:
            parameters = self.get_parameters()
            return SOPf.function_SOP(parameters, self.n)
        
    def get_truth_table(self):
        return RF.truth_table(self.parameters)    

    def get_robustness(self):
        return RF.robustness(self.parameters, self.parameterType)
    
    def assign_parameters(self, parameters=None):
        if parameters is None:
            parameters = ()

        if self.parameterType == 0:
            if len(parameters) != 4:
                raise Exception('Invalid list of hyperparameters. Must have 2 weight matrices and 2 bias terms')
            elif len(parameters) == 0:
                raise Exception('No hyperparameters given in argument.')
            self.parameters = parameters
            self.update_D()
    
        if self.parameterType == 1:
            W1 = PA.uniform_matrix((2**(self.n-1), self.n))
            b1 = PA.assign_b1_from_W1(W1)
            W2 = PA.uniform_matrix((1,2**(self.n-1)))
            b2 = PA.zero_or_one()

            if b2 == 1:
                W2 = -W2

            self.parameters = (W1, b1, W2, b2)
            self.update_D()
            
        if self.parameterType == 2:
            W1 = PA.uniform_matrix((2**(self.n-1), self.n))
            b1 = PA.assign_b1_from_W1(W1)
            W2 = PA.noo_random_matrix((1,2**(self.n-1)))
            b2 = PA.zero_or_one()

            if b2 == 1:
                W2 = -W2

            self.parameters = (W1, b1, W2, b2)
            self.update_D()
            
        if self.parameterType == 3:
            W1 = PA.uniform_matrix((2**(self.n-1), self.n))
            b1 = PA.assign_b1_from_W1(W1)
            W2 = PA.uniform_matrix((1,2**(self.n-1)), positive=True)
            b2 = PA.zero_or_one()

            if b2 == 1:
                W2 = -W2
            
            self.parameters = (W1, b1, W2, b2)
            self.update_D()