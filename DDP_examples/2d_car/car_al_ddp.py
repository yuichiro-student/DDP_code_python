# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 21:36:15 2022

@author: yaoya
"""
import sys
sys.path.insert(0,'../../DDP_code')
sys.path.insert(0,'../../AL_DDP_code')

import numpy as np
from sympy import *
import math
from ddp_cost import F_cost_quadratic, L_cost_quadratic_xu
from car_info import par_dyn, grad_dyn_car, dyn_car, graph_car, graph_car_with_noise
from AL_ddp import options_lagr, AL_ddp
from realization_with_noise import realization_with_noise
from ddp_ctrl_constrained import par_ddp


# initialize system parameters
dt = 0.02
N = 100

n_x = 4         #number of states
n_u = 2         #number of controls


x0 = np.zeros(n_x)  #initial state
xf = np.array([3, 3, math.pi/2, 0])   #desired state

lims_u = np.array([math.pi/2, 7])
lims_l = np.array([-math.pi/2,-7]) #box xontrol constraints
lims = np.array([lims_u, lims_l])
R =  0.05*np.diag(np.array([0.0001, 0.0001])) #running cost weight
K_final = np.diag(np.array([100, 100, 100, 100]))  #terminal cost weight
K = 0*np.diag(np.array([100, 100, 100, 100]))



par_dyn = par_dyn(n_x, n_u, dt, lims)

f_dyn = lambda x, u: dyn_car(x,u,par_dyn)
grad_dyn = lambda x, u: grad_dyn_car(x,u,par_dyn)

L_cost = lambda x, u,k, grad_bool: L_cost_quadratic_xu(x, u, k, R, K, par_dyn, xf, grad_bool)
F_cost = lambda x, grad_bool: F_cost_quadratic(x, K_final, par_dyn, xf, grad_bool)




iter = 80                   #AL outer loop
omega0 = 4                  #inner_ddp tol initial value
delta = 0.98                #inner ddp tol multiplier
beta = 4                    #for changing mu
gamma = 0.4                 #for constratint improvement
mu0 = 0.1                     #initial penalty parameter
lambda_al0 = 0.1            #initial multiplier
con_satisfaction = 1e-6     #constraints satisfaction
rad_con = np.array([0.5,0.5,0.5])
center_con = np.array([[1,0.7],[1,2],[2.5,2]])
num_con = len(rad_con)
min_lambda_al = 1e-15
mu_max = 1e8


x = [symbols('x%d' % i) for i in range(n_x)]
#g = [sympy.symbols('g%d' % i) for i in range(num_con)]
#x = np.asarray(x)


g = [rad_con[0]**2 - (x[0] - center_con[0,0])**2 -(x[1] - center_con[0,1])**2,
     rad_con[1]**2 - (x[0] - center_con[1,0])**2 -(x[1] - center_con[1,1])**2,
     rad_con[2]**2 - (x[0] - center_con[2,0])**2 -(x[1] - center_con[2,1])**2]
    
grad_g = [[-2*(x[0]-center_con[0,0]), -2*(x[1] - center_con[0,1]), 0, 0],
              [-2*(x[0]-center_con[1,0]), -2*(x[1] - center_con[1,1]),0, 0],
              [-2*(x[0]-center_con[2,0]), -2*(x[1] - center_con[2,1]), 0, 0]]

hess = np.block([[-2*np.eye(2),   np.zeros((2, 2))],[np.ones((2, 2)), np.zeros((2,2))]])    
hess_g = np.repeat(hess[:, :, np.newaxis], num_con, axis=2)

g_con = lambdify([x],g,"numpy")
grad_g_con = lambdify([x],grad_g,"numpy")
hess_g_con = lambdify([x],hess_g,"numpy")





par_ddp = par_ddp(f_dyn, grad_dyn, L_cost, F_cost, x0, N, xf)
options_lagr = options_lagr(iter,omega0,beta,gamma,delta,mu0,lambda_al0,con_satisfaction,
                            min_lambda_al,mu_max,rad_con, center_con,g_con,grad_g_con,hess_g_con)
#options for DDP
ddp_iter = 100;          #number of max iterations
cost_tol = 1e-5;         #cost change 1e-3
lambda_reg = 1;          #initial value for lambda
dlambda_reg = 1;         #initial value for dlambda
lambdaFactor = 1.6;      #lambda scaling factor
lambdaMax = 1e10;        #lambda maximum value
lambdaMin = 1e-6;        #below this value lambda = 0
options_ddp = {'ddp_iter': ddp_iter, 'cost_tol': cost_tol, 'lambda_reg': lambda_reg,
               'dlambda_reg': dlambda_reg, 'lambdaFactor': lambdaFactor, 'lambdaMax': lambdaMax,
               'lambdaMin': lambdaMin}




u_bar = np.zeros((n_u,N-1))
#x_ddp, u_ddp, K_out, u_bars, x_bars, J, norm_costgrad = ddp_ctrl_constrained(u_bar,par_ddp,par_dyn,options_ddp)
x_ddp, u_ddp, K_ddp, u_bars, x_bars,cost_out,g_out = AL_ddp(u_bar,par_ddp,par_dyn,options_ddp,options_lagr)
#graph_car(x_ddp,u_ddp,par_dyn,par_ddp,options_lagr)
cov_noise = 0.01*np.array([[math.pi/3,0.01],[0.01, 1]])
x_noise, u_noise, num_violation= realization_with_noise(x_ddp, u_ddp, K_ddp, par_dyn, par_ddp, cov_noise, num_samples = 10)
graph_car_with_noise(x_ddp,u_ddp,x_noise,par_dyn,par_ddp,options_lagr)
