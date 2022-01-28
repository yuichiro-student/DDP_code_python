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
from quad_info import par_dyn, grad_dyn_quad, dyn_quad, graph_quad,graph_quad_with_noise
from AL_ddp import options_lagr, AL_ddp
from realization_with_noise import realization_with_noise
from ddp_ctrl_constrained import par_ddp,ddp_ctrl_constrained
import matplotlib.pyplot as plt

# initialize system parameters
dt = 0.01
N = 200

n_x = 12         #number of states
n_u = 4         #number of controls

m  = 0.468;             #[kg]     mass of quadrotor
Ixx  = 4.856*10**(-3);   #[kgs^2]    inertia of quadrotor
Iyy  = 4.856*10**(-3);   #[kgs^2]    inertia of quadrotor
Izz  = 8.801*10**(-3);   #[kgs^2]    inertia of quadrotor
l = 0.225;              #[m]arm length of rotor;
k = 2.980*10**(-6); #lift constant
b = 1.140*10**(-7); #drag constant
kd  = 0.25;            # [Ns/m]
gravity   = 9.8;             # [m/s^2]  acceleration of gravity


x0 = np.array([0.0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  #initial state
xf = np.array([1.0, 4.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])   #desired state
lims_u = 4*np.ones(n_u)
lims_l = 0*np.ones(n_u) #box xontrol constraints
lims = np.array([lims_u, lims_l])
lims = np.array([])
R =  0.0001*np.diag(np.ones(n_u)) #running cost weight
cost_a, cost_b = 150,50
ones = np.ones(3)
K_final_diag = np.concatenate((cost_a*ones, cost_b*ones, cost_b*ones,cost_b*ones),axis =0 )
K_final = np.diag(K_final_diag)  #terminal cost weight
K = 0*K_final



par_dyn = par_dyn(n_x, n_u, dt, lims, m, Ixx, Iyy, Izz, l, k, b, kd, gravity)
f_dyn = lambda x, u: dyn_quad(x,u,par_dyn)
grad_dyn = lambda x, u: grad_dyn_quad(x,u,par_dyn)
L_cost = lambda x, u,k, grad_bool: L_cost_quadratic_xu(x, u, None, k, R, K, par_dyn, xf, grad_bool)
F_cost = lambda x, grad_bool: F_cost_quadratic(x, K_final, par_dyn, xf, grad_bool)

iter = 80                   #AL outer loop
omega0 = 4                  #inner_ddp tol initial value
delta = 0.98                #inner ddp tol multiplier
beta = 4                    #for changing mu
gamma = 0.4                 #for constratint improvement
mu0 = 0.1                     #initial penalty parameter
lambda_al0 = 0.1            #initial multiplier
con_satisfaction = 1e-6     #constraints satisfaction
rad_con = np.array([1,1,1])
center_con = np.array([[1,1,1],[2,2,3],[4,4,3]])
num_con = len(rad_con)
min_lambda_al = 1e-15
mu_max = 1e8


x = [symbols('x%d' % i) for i in range(n_x)]
#g = [sympy.symbols('g%d' % i) for i in range(num_con)]
#x = np.asarray(x)

g = np.array([rad_con[0]**2 - (x[0] - center_con[0,0])**2 -(x[1] - center_con[0,1])**2 - (x[2]-center_con[0,2])**2,
     rad_con[1]**2 - (x[0] - center_con[1,0])**2 -(x[1] - center_con[1,1])**2 - (x[2]-center_con[1,2])**2,
     rad_con[2]**2 - (x[0] - center_con[2,0])**2 -(x[1] - center_con[2,1])**2 - (x[2]-center_con[2,2])**2])
    
grad_g = np.array([[-2*(x[0]-center_con[0,0]), -2*(x[1] - center_con[0,1]), -2*(x[2] - center_con[0,2]) ],
              [-2*(x[0]-center_con[1,0]), -2*(x[1] - center_con[1,1]), -2*(x[2] - center_con[1,2])],
              [-2*(x[0]-center_con[2,0]), -2*(x[1] - center_con[2,1]), -2*(x[2] - center_con[2,2])]])

grad_g = np.concatenate((grad_g, np.zeros((3,9))),axis = 1)

hess = np.block([[-2*np.eye(3),   np.zeros((3, 9))],[np.ones((9, 3)), np.zeros((9,9))]])    
hess_g = np.repeat(hess[:, :, np.newaxis], 3, axis=2)

g_con = lambdify([x],g,"numpy")
grad_g_con = lambdify([x],grad_g,"numpy")
hess_g_con = lambdify([x],hess_g,"numpy")



par_ddp = par_ddp(f_dyn, grad_dyn, L_cost, F_cost, x0, N, xf)
options_lagr = options_lagr(iter,omega0,beta,gamma,delta,mu0,lambda_al0,con_satisfaction,
                            min_lambda_al,mu_max,rad_con, center_con,g_con,grad_g_con,hess_g_con)
#options for DDP
ddp_iter = 100;          #number of max iterations
cost_tol = 1e-3;         #cost change 1e-3
lambda_reg = 1;          #initial value for lambda
dlambda_reg = 1;         #initial value for dlambda
lambdaFactor = 1.6;      #lambda scaling factor
lambdaMax = 1e10;        #lambda maximum value
lambdaMin = 1e-6;        #below this value lambda = 0
options_ddp = {'ddp_iter': ddp_iter, 'cost_tol': cost_tol, 'lambda_reg': lambda_reg,
               'dlambda_reg': dlambda_reg, 'lambdaFactor': lambdaFactor, 'lambdaMax': lambdaMax,
               'lambdaMin': lambdaMin}




u_bar = np.ones((n_u,N-1))*m*gravity/4
x_ddp, u_ddp, K_out, u_bars, x_bars, J, norm_costgrad = ddp_ctrl_constrained(u_bar,par_ddp,par_dyn,options_ddp)
graph_quad(x_ddp,u_ddp,par_dyn,par_ddp,[])
#x_ddp, u_ddp, K_ddp, u_bars, x_bars,cost_out,g_out = AL_ddp(u_bar,par_ddp,par_dyn,options_ddp,options_lagr)
#graph_quad(x_ddp,u_ddp,par_dyn,par_ddp,options_lagr)
#cov_noise = 0.01*np.array([[4,0,0,0],[0,4,0,0],[0,0,4,0],[0,0,0,4]])
#x_noise, u_noise = realization_with_noise(x_ddp, u_ddp, K_ddp, par_dyn, par_ddp, options_lagr,cov_noise, num_samples = 100)
#graph_quad_with_noise(x_ddp,u_ddp,x_noise,par_dyn,par_ddp,options_lagr,False)
plt.show()
