# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:33:44 2022

@author: yaoya
"""
import numpy as np

def realization_with_noise(x_star,u_star,K,par_dyn,par_ddp,options_lagr,cov_noise,num_samples):
    # put noise to control channel
    # x_star, u_star: reference state and control seq.
    # K: feed back gain     
    # cov_noise: noise covariance matrix
    x_out = np.zeros([par_dyn.n_x, par_ddp.N, num_samples])
    u_out = np.zeros([par_dyn.n_u, par_ddp.N-1, num_samples])
    num_violation = 0
    for i in range(num_samples):
        zero_mean = np.zeros(par_dyn.n_u)
        u_dist = u_star + np.random.multivariate_normal(zero_mean, cov_noise, par_ddp.N-1).T
        u_out[:,:,i] = u_dist.copy()
        x_k = par_ddp.x0
        x_out[:,0,i] = x_k.copy()
        for k in range(par_ddp.N-1):
            u_dist_k = u_dist[:,k].copy();
            u_k = u_dist_k + K[:,:,k]@(x_k- x_star[:,k])
            x_k = par_ddp.f_dyn(x_k, u_k);
            x_out[:,k+1,i] = x_k;
        #check constraint vilation
        G_con = np.asarray(options_lagr.g_con(x_out[:,:,i]))
        G_con_max =  G_con.max()
        if G_con_max>0:
            num_violation = num_violation +1
    print("Number of failed trajectory: %d" % (num_violation))        
    return x_out, u_out, 
    
 