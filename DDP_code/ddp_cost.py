# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 19:14:13 2022

@author: yaoya
"""

import numpy as np

def F_cost_quadratic(x,K_final, par_dyn, xf, grad_bool):    
#terminal cost, quadratic in x
#x is passed in as a raw vector
    if not grad_bool:  #no gradients
        #F = 0.5 * np.matmul(x - xf).T, np.matmul(K_final,(x - xf));
        F = 0.5 * np.dot((x - xf), np.matmul(K_final,(x - xf)))
        return F
    else:
        F_x = K_final@(x - xf);
        F_xx = K_final;
        return F_x, F_xx
    
    
def L_cost_quadratic_xu(x, u, x_ref, k, R, K, par_dyn, xf, grad_bool):
# running cost, quadratic in x and u
# x and y are passed in as raw vectors
    n_x = par_dyn.n_x
    n_u = par_dyn.n_u
    if not grad_bool: #no gradients No reference
        if x_ref is None:
            L = 0.5 * np.dot(u, np.matmul(R,u)) + 0.5*np.dot((x-xf), np.matmul(K,(x-xf)))
        else:
            L = 0.5 * np.dot(u, np.matmul(R,u)) + 0.5*np.dot((x-x_ref[:,k]), np.matmul(K,(x-x_ref[:,k])))
        return L
    else:
        L_u = np.matmul(R,u)
        L_uu = R
        L_xu = np.zeros((n_x, n_u))
        L_xx = K;
        if x_ref is None:
            L_x = np.matmul(K,(x-xf))
        else:
            L_x = np.matmul(K,(x-x_ref[:,k]))
        return L_x, L_u, L_xu, L_uu, L_xx    