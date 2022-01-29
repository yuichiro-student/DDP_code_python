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
import matplotlib.pyplot as plt
import math
from ddp_cost import F_cost_quadratic, L_cost_quadratic_xu
from point_mass_info import dynamics, grad_dynamics, draw_graph, draw_graph_with_noise, ParDyn
from AL_ddp import options_lagr, AL_ddp
from realization_with_noise import realization_with_noise
from ddp_ctrl_constrained import par_ddp


# initialize system parameters
dt = 0.02
N = 100

n_x = 4         #number of states
n_u = 2         #number of controls

#x0 = np.array([[3, 2.75, 0, 0])     #initial state
#xf = np.array([3, 3, 0, 0])         #desired state

lims_u = 50*np.array([1, 1])
lims_l = -15*np.array([-1,-1]) #box xontrol constraints
lims = np.array([lims_u, lims_l])
lims = np.array([])
R = 0.005*1e-4*np.diag(np.array([1, 1])) #running cost weight
K_final = np.diag(np.array([50, 50, 50, 50]))  #terminal cost weight
K = np.diag(np.array([5, 5, 0, 0]))*0

x_ref = np.load('vanilla_ddp_x_star.npy')
x0 = x_ref[:, 0].copy()     #initial state
xf = x_ref[:, -1].copy()    # desired state

par_dyn = ParDyn(n_x, n_u, dt, lims)
f_dyn = lambda x, u: dynamics(x,u,par_dyn)
grad_dyn = lambda x, u: grad_dynamics(x,u,par_dyn)

L_cost = lambda x, u,k, grad_bool: L_cost_quadratic_xu(x, u, x_ref, k, R, K, par_dyn, xf, grad_bool)
F_cost = lambda x, grad_bool: F_cost_quadratic(x, K_final, par_dyn, xf, grad_bool)
# None in L_cost means no trajectory to track



iter = 200                 #AL outer loop
omega0 = 4                  #inner_ddp tol initial value
delta = 0.98                #inner ddp tol multiplier
beta = 4                    #for changing mu
gamma = 0.4                 #for constratint improvement
mu0 = 0.1                     #initial penalty parameter
lambda_al0 = 0.001            #initial multiplier
con_satisfaction = 1e-6     #constraints satisfaction
rad_con = np.array([1])
center_con = np.array([[2, 2]])
num_con = len(rad_con)
min_lambda_al = 1e-15
mu_max = 10


x = [symbols('x%d' % i) for i in range(n_x)]
#g = [sympy.symbols('g%d' % i) for i in range(num_con)]
#x = np.asarray(x)
g = [rad_con[0] ** 2 - (x[0] - center_con[0, 0]) ** 2 - (x[1] - center_con[0, 1]) ** 2]

grad_g = [[-2 * (x[0] - center_con[0, 0]), -2 * (x[1] - center_con[0, 1]), 0,0  ]]

hess = np.block([[-2 * np.eye(2), np.zeros((2, 2))], [np.ones((2, 2)), np.zeros((2, 2))]])
hess_g = np.repeat(hess[:, :, np.newaxis], 1, axis=2)

g_con = lambdify([x],g,"numpy")
grad_g_con = lambdify([x],grad_g,"numpy")
hess_g_con = lambdify([x],hess_g,"numpy")





par_ddp = par_ddp(f_dyn, grad_dyn, L_cost, F_cost, x0, N, xf)
options_lagr = options_lagr(iter,omega0,beta,gamma,delta,mu0,lambda_al0,con_satisfaction,
                            min_lambda_al,mu_max,rad_con, center_con,g_con,grad_g_con,hess_g_con)
#options for DDP
ddp_iter = 80;          #number of max iterations
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
#x_ddp, u_ddp, K_ddp, u_bars, x_bars,cost_out,g_out = \
#    AL_ddp(u_bar, par_ddp, par_dyn, options_ddp, options_lagr, boxQP_flag = False)
#draw_graph(x_ddp,u_ddp,par_dyn,par_ddp,options_lagr)

x_ref = np.load('vanilla_ddp_x_star.npy')
cov_noise = 100*np.array([[1,0],[0, 1]])
flag = 1
if flag == 0:
    x_tmp = np.load('BS/point_mass_X.npy')
    u_ddp = np.load('BS/point_mass_U.npy')
    K_tmp = np.load('BS/point_mass_K.npy')
    K_ddp = K_tmp[:,0:4,:].copy()
    x_ddp = x_tmp[0:4,:].copy()
    draw_graph(x_ddp,u_ddp,x_ref,par_dyn,par_ddp,options_lagr)
    plt.show()
    plt.savefig('CBF_track.eps', format='eps')
    x_noise, u_noise = realization_with_noise(x_ddp, u_ddp, K_ddp, par_dyn, par_ddp, options_lagr,cov_noise, num_samples = 500)
    draw_graph_with_noise(x_ddp,u_ddp,x_ref,x_noise,par_dyn,par_ddp,options_lagr)
    plt.show()
    plt.savefig('CBF_noise.eps', format='eps')
if flag == 1:
    x_ddp = np.load('AL/AL_DDP_track_x.npy')
    u_ddp = np.load('BS/point_mass_U.npy')
    x_noise = np.load('AL/AL_DDP_track_x_noise.npy')
    draw_graph(x_ddp, u_ddp, x_ref, par_dyn, par_ddp, options_lagr)
    plt.show()
    plt.savefig('AL_track.eps', format='eps')
   # draw_graph_with_noise(x_ddp, u_ddp, x_ref, x_noise, par_dyn, par_ddp, options_lagr)
   # plt.show()
   # plt.savefig('AL_noise.eps', format='eps')
if flag == 2:
    x_ddp = np.load('ContBF/cnt_track_x.npy').T
    u_ddp = np.load('BS/point_mass_U.npy')
    x_noise = np.load('ContBF/cnt_track_x_noise.npy').transpose((1,0,2))
    draw_graph(x_ddp, u_ddp, x_ref, par_dyn, par_ddp, options_lagr)
    plt.show()
    plt.savefig('CntBF_track_x.eps', format='eps')
    draw_graph_with_noise(x_ddp, u_ddp, x_ref, x_noise, par_dyn, par_ddp, options_lagr)
    plt.show()
    plt.savefig('CntBF_track_x_noise.eps', format='eps')