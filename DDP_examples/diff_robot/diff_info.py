# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 21:28:22 2022

@author: yaoya
"""
import math
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

#for plotting ellipsoids
from scipy.stats import norm
from scipy.stats.distributions import chi2


class par_dyn:
    def __init__(self, n_x, n_u, dt, lims,d,rl,rr):
        self.n_x  = n_x
        self.n_u  = n_u
        self.dt = dt
        self.lims = lims
        self.d = d
        self.rl = rl
        self.rr = rr
        

def grad_dyn_diff(x,u,dyn):
    
    N_x = x.shape[0]
    dt = dyn.dt
    d = dyn.d
    r2 = dyn.rr
    r1 = dyn.rl
    v = (r2*u[1]+r1*u[0])/2;
    temp = np.array([[0, 0, -v*math.sin(x[2])], [0, 0, v*math.cos(x[2])],[0, 0, 0]])
    fx = np.eye(N_x)+temp*dt;
    fu = np.array([[r1*math.cos(x[2])/2,r2*math.cos(x[2])/2],
                   [r1*math.sin(x[2])/2,r2*math.sin(x[2])/2],
                   [-r1/(2*d), r2/(2*d)]])*dt;
    return fx, fu

def dyn_diff(x,u,dyn):
    dt = dyn.dt
    d = dyn.d
    r2 = dyn.rr
    r1 = dyn.rl
    v = (r2*u[1]+r1*u[0])/2;
    omega = (r2*u[1]-r1*u[0])/(2*d);
    
    
    xdot = v*math.cos(x[2]);
    ydot = v*math.sin(x[2]);
    thetadot = omega;

    y = x + np.array([xdot, ydot,thetadot])*dt
    return y

def plot_target_and_initial_pnt(x,y,par_ddp,ax1,sz):
    ax1.scatter(par_ddp.xf[0],par_ddp.xf[1],c = "red",s = 2*sz)
    ax1.scatter(x[0], y[0], c = "blue",s = sz)
    return 0
        
def plot_trj(x,y,theta,par_ddp,ax1,sz,line_color, linestyle, line_width):
    ax1.plot(x,y,color = line_color,linewidth = line_width,linestyle = linestyle)
    ax1.scatter(par_ddp.xf[0],par_ddp.xf[1],c = "red",s = 2*sz)
    ax1.scatter(x[0], y[0], c = "blue",s = sz)
    ax1.scatter(x[-1], y[-1], c = "green",s = sz)    
    ax1.quiver(x[0], y[0], math.cos(theta[0]), math.sin(theta[0]), scale=30)
    ax1.quiver(x[-1], y[-1], math.cos(theta[-1]), math.sin(theta[-1]),scale=30,color = "green")
    return 0
    
def plot_obs(options_lagr,ax1):
    rad_con = options_lagr.rad_con
    center_con = options_lagr.center_con
    num_con = options_lagr.num_con
    for i in range(num_con):
        circle = plt.Circle(center_con[i,:],rad_con[i], color = 'gray')
        ax1.add_patch(circle)
    return 0

def graph_diff(x_ddp,u_ddp,x_ref,par_dyn,par_ddp,options_lagr):
    N = par_ddp.N
    sz = 40
    x = x_ddp[0,:]
    y = x_ddp[1,:]
    theta = x_ddp[2,:]
    fig, ax1 = plt.subplots()
    plot_target_and_initial_pnt(x, y, par_ddp, ax1,sz)  
    if options_lagr:
        plot_obs(options_lagr, ax1)
    plot_trj(x, y, theta, par_ddp, ax1, sz,'blue', 'solid', 2)
    if not x_ref.shape[0] == 0:
        plot_trj(x_ref[0,:], x_ref[1,:], theta, par_ddp, ax1, sz,'orangered', 'dashed',2)
    ax1.set_aspect('equal', 'box')
    utime = par_dyn.dt*np.linspace(0,N-2,N-1)#0 to N-2
    
    #plt.figure()
    #plt.plot(utime,u_ddp[0,:],c = "blue")
    #plt.plot(utime,u_ddp[1,:],c = "red")
    #plt.xlabel("Time[s]")
    #plt.ylabel("control[rad/s]")
    return 0
    
def graph_diff_with_noise(x_ddp,u_ddp,x_noise,par_dyn,par_ddp,options_lagr):
    N = par_ddp.N
    sz = 40
    x_ref = x_ddp[0,:]
    y_ref = x_ddp[1,:]
    theta_ref = x_ddp[2,:]
    fig, ax1 = plt.subplots()
    plot_target_and_initial_pnt(x_ref, y_ref, par_ddp, ax1,sz)  
    if options_lagr:
        plot_obs(options_lagr, ax1)
    num_trj = x_noise.shape[2]
    for i in range(num_trj):
        x = x_noise[0,:,i]
        y = x_noise[1,:,i]
        theta = x_noise[2,:,i]
        plot_trj(x, y, theta, par_ddp, ax1, sz,'red','solid',0.5)
    plot_trj(x_ref, y_ref, theta_ref, par_ddp, ax1, sz,'blue','solid',2)
    ax1.set_aspect('equal', 'box')
    
    STD = 3 #scaling paramter of normal standard div
    conf = 2*norm.cdf(STD)-1 #double tail
    scale = chi2.ppf(conf, df=2)
    # averaging ove num_of_trj to get sequence of mu
    Mu = np.mean(x_noise[0:2,:,:], axis = 2)
    # get state trajectory with noise at each time step
    t = np.linspace( 0 , 2 * np.pi , 150 )
    for k in range(N):
        x_k = x_noise[0:2,k,:].copy() #nx(only pos) by num_trj
        cov = np.asarray(np.cov(x_k))
        D,V = np.linalg.eig(cov*scale) # d eigenvalue, v is eigen matrix
        order = np.flip(np.argsort(D)) #large to small
        D = np.diag(D)
        V = V[:,order]
        VV = V@np.sqrt(D)
        e = np.array([np.cos(t),np.sin(t)])
        Mu_k = Mu[:,k]
        e2 = VV@e + Mu_k[:,None]
        ax1.plot(e2[0,:], e2[1,:],c = "black",linewidth = 0.5)   

    utime = par_dyn.dt*np.linspace(0,N-2,N-1) # 0 to N-2
    #plt.figure()
    #plt.plot(utime,u_ddp[0,:],c = "blue")
    #plt.plot(utime,u_ddp[1,:],c = "red")
    #plt.xlabel("Time[s]")
    #plt.ylabel("control[rad/s]")
    return 0
    
    
    
    