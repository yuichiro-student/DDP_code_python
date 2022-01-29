# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 21:28:22 2022

@author: yaoya
"""
import math
import numpy as np
from sympy import *
import matplotlib.pyplot as plt


class ParDyn:
    def __init__(self, n_x, n_u, dt, lims):
        self.n_x  = n_x
        self.n_u  = n_u
        self.dt = dt
        self.lims = lims


def grad_dynamics(x,u,dyn):
    n_x = x.shape[0]
    dt = dyn.dt
    fx = np.eye(n_x)+np.array([[0., 0., 1., 0.],
                               [0., 0., 0., 1.],
                               [0., 0., 0., 0.], [0., 0., 0., 0.]])*dt
    fu = np.array([[0, 0], [0, 0], [1., 0], [0., 1]])*dt
    return fx, fu


def dynamics(x,u,dyn):
    dt = dyn.dt
    dx = np.array([x[2], x[3], u[0], u[1]])*dt
    y = x + dx
    return y


def plot_target_and_initial_pnt(x,y,par_ddp,ax1,sz):
    ax1.scatter(par_ddp.xf[0],par_ddp.xf[1],c = "black",s = 8*sz, marker = '*')
    ax1.scatter(x[0], y[0], c = "blue",s = 4*sz, marker = 'X')
    return 0


def plot_trj(x, y, theta,par_ddp, ax1,sz,line_color,linestyle, line_width):
    if line_color is None:
    # plot line
        ax1.scatter(x[-1], y[-1], s=sz/5)
        ax1.plot(x, y, linewidth=line_width,linestyle = linestyle)
    else:
        ax1.scatter(x[-1], y[-1], c=line_color, s = sz)
        ax1.plot(x,y,color = line_color, linewidth = line_width,linestyle = linestyle)
    # plot initial point with X
    # ax1.scatter(x[0], y[0], c = line_color,s = 2*sz, marker='X')
    # plot final point with circle
    #ax1.scatter(x[-1], y[-1], c = line_color, s = 2*sz)
    # ax1.quiver(x[0], y[0], math.cos(theta[0]), math.sin(theta[0]), scale=30)
    # ax1.quiver(x[-1], y[-1], math.cos(theta[-1]), math.sin(theta[-1]),scale=30,color = "green")
    return 0


def plot_obs(options_lagr, ax1):
    rad_con = options_lagr.rad_con
    center_con = options_lagr.center_con
    num_con = options_lagr.num_con
    for i in range(num_con):
        circle = plt.Circle(center_con[i,:],rad_con[i], color = 'gray')
        ax1.add_patch(circle)
    return 0
    

def draw_graph(x_ddp,u_ddp,x_ref,par_dyn,par_ddp,options_lagr):
    N = par_ddp.N
    sz = 40
    x = x_ddp[0,:]
    y = x_ddp[1,:]
    theta = x_ddp[2,:]
    fig, ax1 = plt.subplots()
    plot_target_and_initial_pnt(x, y, par_ddp, ax1,sz)  
    if options_lagr:
        plot_obs(options_lagr, ax1)
    plot_trj(x, y, theta, par_ddp, ax1, sz,'blue','solid',2)
    if not x_ref.shape[0] == 0:
        plot_trj(x_ref[0,:], x_ref[1,:], theta, par_ddp, ax1, sz,'orangered', 'dashed',2)
    ax1.set_aspect('equal', 'box')
    hfont = {'fontname': 'Helvetica'}
    utime = par_dyn.dt*np.linspace(0,N-2,N-1)#0 to N-2
    ax1.set_xlabel('x', **hfont)
    ax1.set_ylabel('y', **hfont)


    #plt.figure()
    #plt.plot(utime,u_ddp[0,:],c = "blue")
    #plt.plot(utime,u_ddp[1,:],c = "red")
    #plt.xlabel("Time[s]")
    #plt.ylabel("control[rad/s]")
    return 0

def draw_graph_with_noise(x_ddp,u_ddp,x_org,x_noise,par_dyn,par_ddp,options_lagr):
    N = par_ddp.N
    sz = 40
    x_ref = x_ddp[0,:]
    y_ref = x_ddp[1,:]
    theta_ref = x_ddp[2,:]
    fig, ax1 = plt.subplots()
    num_trj = x_noise.shape[2]
    for i in range(num_trj):
        x = x_noise[0, :, i]
        y = x_noise[1, :, i]
        theta = x_noise[2, :, i]
        plot_trj(x, y, theta, par_ddp, ax1, sz, None, 'solid', 0.5)

    plot_target_and_initial_pnt(x_ref, y_ref, par_ddp, ax1,sz)
    plot_trj(x_org[0, :], x_org[1, :], x_org[2,:],par_ddp, ax1, sz, 'orangered', 'dashed', 2)


    if options_lagr:
        plot_obs(options_lagr, ax1)

    #plot_trj(x_ref, y_ref, theta_ref, par_ddp, ax1, sz,'blue','solid',2)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    hfont = {'fontname': 'Helvetica'}
    utime = par_dyn.dt * np.linspace(0, N - 2, N - 1)  # 0 to N-2
    ax1.set_xlabel('x', **hfont)
    ax1.set_ylabel('y', **hfont)
    #utime = par_dyn.dt*np.linspace(0,N-2,N-1)#0 to N-2
    #plt.figure()
    #plt.plot(utime,u_ddp[0,:],c = "blue")
    #plt.plot(utime,u_ddp[1,:],c = "red")
    #plt.xlabel("Time[s]")
    #plt.ylabel("control[rad/s]")
    return 0
    
    