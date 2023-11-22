import numpy as np
from sympy.logic import SOPform, boolalg
from sympy import symbols, srepr, Not
import re

class BoolFunc():
    def __init__(self, params=[], n=3, param_type=0):
        self.n = n   # input size
        self.param_type = param_type
        self.assign_params(params)
            
    def relu(self, x):
        return x * (x>0)
    
    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2]
    
    def get_param_type(self):
        return self.param_type
    
    def change_param_type(self, x):
        if x >= 4:
            raise Exception('param_type must be a number between 0 and 3.')
        self.param_type = x
        self.assign_params()
        print(f'Neural network changed to type {x}')
    
    def forward(self, x):
        x = self.W1@x + self.b1
        x = self.relu(x)
        x = self.W2@x + self.b2
        x = self.relu(x)
        return x
    
    def update_D(self):
        if self.param_type == 0:
            raise Exception('Not supported yet')
        
        if self.param_type == 1:    
            logic_expr = self.get_SOP()
            force = False if self.n < 8 else True
            logic_expr_conj = boolalg.to_dnf(Not(logic_expr), simplify=True, force=force)
            str_le = srepr(logic_expr)
            str_le_conj = srepr(logic_expr_conj)
            
            # Find number of literals in both expression
            num_lit = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('Symbol'), str_le))
            num_lit_conj = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('Symbol'), str_le_conj))
        
            if num_lit < num_lit_conj:
                self.D = num_lit
            else:
                selfD = num_lit_conj
        
            self.D = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('Symbol'), str_le)) + self.b2  # number of literals plus value of b2
        
        if self.param_type == 2:
            raise Exception('Not supported yet')
        
        if self.param_type == 3:    
            logic_expr = self.get_SOP()
            force = False if self.n < 8 else True
            logic_expr_conj = boolalg.to_dnf(Not(logic_expr), simplify=True, force=force)
            str_le = srepr(logic_expr)
            str_le_conj = srepr(logic_expr_conj)
            
            # Find number of literals in both expression
            num_lit = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('Symbol'), str_le))
            num_lit_conj = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape('Symbol'), str_le_conj))
            
            # Choose the expression that minimises the number of literals
            if num_lit < num_lit_conj:
                if logic_expr == True or logic_expr == False:
                    self.D = 0

                elif not isinstance(logic_expr, boolalg.Or):
                    num_clau = 1
                    self.D = num_clau + num_lit

                else:
                    num_clau = len(logic_expr.args)   # number of clauses
                    self.D = num_clau + num_lit
            else:
                if logic_expr_conj == True or logic_expr_conj == False:
                    self.D = 0

                elif not isinstance(logic_expr_conj, boolalg.Or):
                    num_clau_conj = 1
                    self.D = num_clau_conj + num_lit_conj

                else:
                    num_clau_conj = len(logic_expr_conj.args)   # number of clauses
                    self.D = num_clau_conj + num_lit_conj
        
    def get_D(self):
        return self.D
    
    def get_SOP(self):
        if self.param_type == 0:
            raise Exception('Not supported yet.')
        
        if self.param_type == 1 or self.param_type == 2 or self.param_type == 3:
            sym = ''
            for i in range(self.n):
                sym = sym+str(i)+' '
            sym_list = list(symbols(sym))

            minterms = []
            for i in range(2**(self.n-1)):
                if ((-1 in self.W1[i,:]) or (1 in self.W1[i,:])) and self.W2[0,i] != 0:
                    minterms.append({})
                    for j in range(self.n):
                        if self.W1[i,j] == 1:
                            minterms[-1][sym_list[j]] = 1
                        elif self.W1[i,j] == -1:
                            minterms[-1][sym_list[j]] = 0
                        else:
                            pass

            if self.b2 == 0:
                return SOPform(sym_list, minterms)
            elif self.b2 == 1:
                force = False if self.n < 8 else True
                return boolalg.to_dnf(Not(SOPform(sym_list, minterms)), simplify=True, force=force)
            else:
                raise Exception('b2 has no valid value, must be 0 or 1.')
            
                            
    
    def assign_params(self, params=[]):
        if self.param_type == 0:
            if len(params) != 4:
                raise Exception('Invalid list of hyperparameters. Must have 2 weight matrices and 2 bias terms')
            elif len(params) == 0:
                raise Exception('No hyperparameters given in argument.')
            self.W1 = np.array(params[0])
            self.b1 = np.array(params[1])
            self.W2 = np.array(params[2])
            self.b2 = np.array(params[3])
            self.update_SOP_and_D()
    
        if self.param_type == 1:
            self.W1 = np.random.randint(-1, 2, size=(2**(self.n-1), self.n))
            
            self.b1 = np.zeros((2**(self.n-1),1), dtype=int)
            for i in range(2**(self.n-1)):
                if (-1 in self.W1[i,:]) or (1 in self.W1[i,:]):
                    freq_1 = 0
                    for j in self.W1[i,:]:
                        if j == 1:
                            freq_1 += 1
                    self.b1[i,0] = 1 - freq_1
                else:
                    self.b1[i,0] = 0
                
            self.W2 = np.zeros((1,2**(self.n-1)), dtype=int)
            self.b2 = 0
            if np.random.rand() > 0.5:
                for i in range(2**(self.n-1)):
                    self.W2[0,i] = (-1 in self.W1[i,:]) or (1 in self.W1[i,:])
            else:
                for i in range(2**(self.n-1)):
                    self.W2[0,i] = -1 * ((-1 in self.W1[i,:]) or (1 in self.W1[i,:]))
                self.b2 = 1
            
            self.update_D()
            
            
        if self.param_type == 2:
            self.W1 = np.random.randint(-1, 2, size=(2**(self.n-1), self.n))
            
            self.b1 = np.zeros((2**(self.n-1),1), dtype=int)
            for i in range(2**(self.n-1)):
                if (-1 in self.W1[i,:]) or (1 in self.W1[i,:]):
                    freq_1 = 0
                    for j in self.W1[i,:]:
                        if j == 1:
                            freq_1 += 1
                    self.b1[i,0] = 1 - freq_1
                else:
                    self.b1[i,0] = 0
                
            self.W2 = np.random.randint(-1, 2, (1,2**(self.n-1)))
            
            self.b2 = 0
            if np.random.rand() > 0.5:
                pass
            else:
                self.b2 = 1
            
            self.update_D()
            
        if self.param_type == 3:
            self.W1 = np.random.randint(-1, 2, size=(2**(self.n-1), self.n))
            
            self.b1 = np.zeros((2**(self.n-1),1), dtype=int)
            for i in range(2**(self.n-1)):
                if (-1 in self.W1[i,:]) or (1 in self.W1[i,:]):
                    freq_1 = 0
                    for j in self.W1[i,:]:
                        if j == 1:
                            freq_1 += 1
                    self.b1[i,0] = 1 - freq_1
                else:
                    self.b1[i,0] = 0
                
            self.W2 = np.random.randint(0, 2, (1,2**(self.n-1)))
            self.b2 = 0
            
            if np.random.rand() > 0.5:
                pass
            else:
                self.W2 = -self.W2
                self.b2 = 1
                
            self.update_D()