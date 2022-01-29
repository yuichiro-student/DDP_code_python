# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 21:23:34 2022

@author: yaoya
"""
import numpy as np
def total_penalty_cost_L(L_cost_fun, penalty_fun, x, u, k, grad_bool, lambda_al, mu, par_dyn, options_lagr):
# total running cost and its gradients with penalty functions
    if not grad_bool: 
        L_cost = L_cost_fun(x, u,k,grad_bool);
        R = penalty_fun(x, u,  lambda_al[:,k], mu[:,k], par_dyn, options_lagr, grad_bool);
        L = L_cost + R
        return L
    else:
        L_c_x, L_c_u, L_c_xu, L_c_uu, L_c_xx = L_cost_fun(x, u, k, grad_bool)
        R_x, R_u, R_xu, R_uu, R_xx= penalty_fun(x, u,  lambda_al[:, k], mu[:, k], par_dyn, options_lagr, grad_bool)
        L_x = L_c_x + R_x
        L_u = L_c_u + R_u
        L_xu = L_c_xu + R_xu
        L_uu = L_c_uu + R_uu
        L_xx = L_c_xx + R_xx
        return L_x, L_u, L_xu, L_uu, L_xx         
        
def total_penalty_cost_F(F_cost_fun, penalty_fun, x, grad_bool, lambda_al, mu, par_dyn, options_lagr):
# total running cost and its gradients with penalty functions
    if not grad_bool:
        F_cost = F_cost_fun(x, grad_bool);
        R = penalty_fun(x, np.zeros(par_dyn.n_u), lambda_al, mu, par_dyn, options_lagr, grad_bool);
        return F_cost + R
    else:        
        F_cost_x, F_cost_xx = F_cost_fun(x, grad_bool);
        R_x, _, _, _, R_xx = penalty_fun(x, np.zeros(par_dyn.n_u), lambda_al, mu, par_dyn, options_lagr, grad_bool);
        F_x = F_cost_x + R_x;
        F_xx = F_cost_xx + R_xx;
        return F_x, F_xx


def penalty_fun(x, u, lambda_al, mu, par_dyn, options_lagr, grad_bool):
# penalty lagrangean function for lagrangian and its derivatives
# only for states here
# to be evaluated for each (applicable) time instant
    n_X = len(x); 
    n_u = par_dyn.n_u
    if not grad_bool:  # no gradients
        R = 0;
        g_con = options_lagr.g_con(x);
        for i in range(options_lagr.num_con):
            R = R + mu[i]/2*max(lambda_al[i]/mu[i]+g_con[i],0)**2
        return R
    else:
        R_u = np.zeros(n_u);
        R_uu = np.zeros((n_u, n_u));
        R_xu = np.zeros((n_X, n_u));
        R_x = np.zeros(n_X);
        R_xx = np.zeros((n_X, n_X));
        g_con = options_lagr.g_con(x);
        grad_g_con = np.asarray(options_lagr.grad_g_con(x));
        hess_g_con = np.asarray(options_lagr.hess_g_con(x));
        for i in range(options_lagr.num_con):
            val = lambda_al[i]+mu[i]*g_con[i]
            R_x = R_x + max(val,0) * grad_g_con[i,:]
            if val > 0:
                R_xx = R_xx + mu[i] * np.matmul(grad_g_con[i,:].T,grad_g_con[i,:]) + val * hess_g_con[:,:,i];
    return R_x, R_u, R_xu, R_uu, R_xx    
