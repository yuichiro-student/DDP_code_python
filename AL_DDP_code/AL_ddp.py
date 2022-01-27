# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:50:45 2022

@author: yaoya
"""
from ddp_ctrl_constrained import ddp_ctrl_constrained
from AL_ddp_cost import total_penalty_cost_F, total_penalty_cost_L, penalty_fun
import numpy as np
from copy import deepcopy


class options_lagr:
    def __init__(self,iter,omega0,beta,gamma,delta,mu0,lambda_al0,con_satisfaction, 
                 min_lambda_al,mu_max, rad_con, center_con,g_con,grad_g_con,hess_g_con):
        self.iter = iter
        self.omega0 = omega0
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mu0 = mu0
        self.num_con = len(rad_con)
        self.lambda_al0 = lambda_al0
        self.con_satisfaction = con_satisfaction
        self.g_con = g_con
        self.grad_g_con = grad_g_con
        self.hess_g_con = hess_g_con
        self.min_lambda_al = min_lambda_al
        self.mu_max = mu_max
        self.rad_con = rad_con
        self.center_con = center_con



def evaluate_actual_cost_constraints(x_traj, u_traj,g_con,par_ddp):
#evaluates constraint functions along trajectories
    G = np.asarray(g_con(x_traj))
#evaluates actual cost along trajectories
    J = 0;
    for k in range(par_ddp.N-1):#0 to N-2
        J = J + par_ddp.L_cost(x_traj[:, k], u_traj[:, k], k,False);
    J = J + par_ddp.F_cost(x_traj[:, -1], False);
    return G, J

def forward_dyn(u_bar,par_dyn,par_ddp):
    x_bar = np.zeros((par_dyn.n_x, par_ddp.N))
    x_bar[:,0] = par_ddp.x0
    for k in range(par_ddp.N-1):
        x_bar[:,k+1] = par_ddp.f_dyn(x_bar[:,k], u_bar[:,k]);
    return x_bar 
    
def update_multipliers(lambda_al, mu, g):
    lambda_new = np.maximum(0, lambda_al + np.multiply(mu,g));
    return lambda_new

def AL_ddp(u_bar,par_ddp,par_dyn,options_ddp,options_lagr, boxQP_flag = True):
# augmented lagrangian on ddp with exponential barrier functions
# setting initial inner parameters and costs for unconstrained ddp
#if smooth flag is true, use smooth approximation of PHR, else, use PHR.
    options_ddp_inner = options_ddp.copy();
    options_ddp_inner['cost_tol'] = options_lagr.omega0;
    par_ddp_inner = deepcopy(par_ddp)
    g_con = options_lagr.g_con;
    lambda_al = options_lagr.lambda_al0 * np.ones((options_lagr.num_con,par_ddp.N));  #initial lagrange multipliers
    mu = options_lagr.mu0 * np.ones((options_lagr.num_con,par_ddp.N));  # initial penalty parameters
    pen_fun = lambda x, u, mu, c, par_dyn, options_lagr, grad_bool: penalty_fun(x, u, mu, c, par_dyn, options_lagr, grad_bool);
    u_bars = np.zeros((par_dyn.n_u, par_ddp.N-1,options_lagr.iter+1))
    x_bars = np.zeros((par_dyn.n_x, par_ddp.N,options_lagr.iter+1))
    g_out = np.zeros(options_lagr.iter+1)
    cost_out = np.zeros(options_lagr.iter+1)
    
    #initial cost and constraint violation
    x_bar_tmp = forward_dyn(u_bar,par_dyn,par_ddp);
    gtmp, cost_out[0] = evaluate_actual_cost_constraints(x_bar_tmp,u_bar,g_con,par_ddp)
    gtmp = np.asarray(gtmp)
    g_out[0] = gtmp.max()
    u_bars[:,:,0] = u_bar.copy()
    x_bars[:,:,0] = x_bar_tmp.copy()
    G_con_old = gtmp;
    

#augmented lagrangian with ddp
    for j in range(options_lagr.iter):
        print('Outer iteration %d ----------------------------------------------' % (j))
        #update cost functions for DDP
        par_ddp_inner.L_cost = lambda x, u,k,grad_bool: total_penalty_cost_L(par_ddp.L_cost, pen_fun,
                                                                           x, u,k, grad_bool, lambda_al, mu, par_dyn, options_lagr)
        par_ddp_inner.F_cost = lambda x, grad_bool: total_penalty_cost_F(par_ddp.F_cost, pen_fun, 
                                                                         x, grad_bool, lambda_al[-1,:], mu[-1,:], par_dyn, options_lagr)

    #solve unconstrained penalty problem
        print('DDP on penalty terms:')
        x_ddp, u_ddp, S_ddp, _, _, _, norm_costgrad = ddp_ctrl_constrained(u_bar, par_ddp_inner, par_dyn, options_ddp_inner, boxQP_flag)
        print('----------------------------')
        G_con, J_actual = evaluate_actual_cost_constraints(x_ddp,u_ddp,g_con,par_ddp)
        G_con_max = G_con.max()
        print('Actual cost after DDP: %f, Max constraint: %f\n' % (J_actual, G_con_max))
        #check convergence criteria
        if G_con_max <= options_lagr.con_satisfaction and norm_costgrad <= options_ddp['cost_tol']:
            if j>4:
                print('norm_grad %f' % norm_costgrad);
                print('options_ddp.convtol_star %f' % (options_ddp["cost_tol"]));
                print('options_ddp_inner.convtol_star %f' % (options_ddp_inner["cost_tol"]));
                print('Convergence of augmented Lagrangian');
                cost_out[j] = J_actual;
                g_out[j] = G_con_max
            break
        
        # update multipliers and penalty terms
        ind_constraintviolation =  G_con > options_lagr.con_satisfaction
        lambda_al = update_multipliers(lambda_al, mu, G_con)
        lambda_al = np.maximum(lambda_al, options_lagr.min_lambda_al * np.ones(lambda_al.shape));
    
    
        if j > 0 and not(len(ind_constraintviolation) == 0) and any(G_con[ind_constraintviolation] > options_lagr.gamma * G_con_old[ind_constraintviolation]):
            print('------- Not enough constraint improvement, increasing penalty parameter')
            
            mu[ind_constraintviolation] = np.minimum(options_lagr.beta * mu[ind_constraintviolation],
                                                 options_lagr.mu_max)
        else:
            print('------- Enough constraint improvement')
            #decrease convergence criterion for DDP
            options_ddp_inner['cost_tol'] = max(options_ddp_inner['cost_tol'] * options_lagr.delta, options_ddp['cost_tol']); 

        print(j)
        u_bar = u_ddp;
        x_bar = x_ddp;
        u_bars[:,:,j+1] = u_bar.copy()
        x_bars[:,:,j+1] = x_bar.copy()
        cost_out[j+1] = J_actual;
        g_out[j+1] = G_con_max;
    

    cost_out = cost_out[0:j+1];
    g_out = g_out[0:j+1];
    u_bars = u_bars[:, :, 0:j];
    x_bars = x_bars[:, :, 0:j];
    return x_ddp, u_ddp, S_ddp, u_bars, x_bars,cost_out,g_out







