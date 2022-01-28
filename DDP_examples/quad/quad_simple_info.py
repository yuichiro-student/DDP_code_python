# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 21:28:22 2022

@author: yaoya
"""
import math
import numpy as np
from sympy import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

#for plotting ellipsoids
from scipy.stats import norm
from scipy.stats.distributions import chi2

class par_dyn:
    def __init__(self, n_x, n_u, dt, lims,m, Ixx, Iyy, Izz, g):
        self.n_x  = n_x
        self.n_u  = n_u
        self.dt = dt
        self.lims = lims
        self.m = m
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = g
        
def grad_dyn_quad(x,u,dyn):
    state = x
    control = u
    n_x = dyn.n_x
    n_u = dyn.n_u
    dt = dyn.dt
    g = dyn.g

    phi = state[0]
    theta = state[1]
    psi = state[2]
    phi_rate = state[3]
    theta_rate = state[4]
    psi_rate = state[5]
    vx = state[6]
    vy = state[7]
    vz = state[8]
    g = dyn.g
    mass = dyn.m
    Ix = dyn.Ixx
    Iy = dyn.Iyy
    Iz = dyn.Izz


    
    ft = control[0] # thrust in body frame



    fx = np.eye(n_x) + dt * np.array([[theta_rate * np.cos(phi) * np.tan(theta) - psi_rate * np.sin(phi) * np.tan(theta), psi_rate * np.cos(phi) * (np.tan(theta) ** 2 + 1) + theta_rate * np.sin(phi) * (np.tan(theta) ** 2 + 1), 
                        0, 1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta), 0, 0, 0, 0, 0, 0],
                       [-psi_rate * np.cos(phi) - theta_rate * np.sin(phi), 0, 0, 0,np.cos(phi), -np.sin(phi), 0, 0, 0, 0, 0, 0],
                       [(theta_rate * np.cos(phi)) / np.cos(theta) - (psi_rate * np.sin(phi)) / np.cos(theta), (psi_rate * np.cos(phi) * np.sin(theta)) / np.cos(theta) ** 2 + (theta_rate * np.sin(phi) * np.sin(theta)) / np.cos(theta) ** 2,
                        0, 0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta), 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, (psi_rate * (Iy - Iz)) / Ix, (theta_rate * (Iy - Iz)) / Ix, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -(psi_rate * (Ix - Iz)) / Iy, 0, -(phi_rate * (Ix - Iz)) / Iy, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, (theta_rate * (Ix - Iy)) / Iz, (phi_rate * (Ix - Iy)) / Iz, 0, 0, 0, 0, 0, 0, 0],
                       [0, -g * np.cos(theta), 0, 0, -vz, vy, 0, psi_rate, -theta_rate, 0, 0, 0],
                       [g * np.cos(phi) * np.cos(theta), -g * np.sin(phi) * np.sin(theta), 0, vz, 0, -vx, -psi_rate, 0, phi_rate, 0, 0, 0],
                       [-g * np.cos(theta) * np.sin(phi), -g * np.cos(phi) * np.sin(theta), 0, -vy, vx, 0, theta_rate, -phi_rate, 0, 0, 0, 0],
                       [vy * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) + vz * (np.cos(phi) * np.sin(psi) - np.cos(psi) * np.sin(phi) * np.sin(theta)),
                        vz * np.cos(phi) * np.cos(theta) * np.cos(psi) - vx * np.cos(psi) * np.sin(theta) + vy * np.cos(theta) * np.cos(psi) * np.sin(phi), vz * (np.cos(psi) * np.sin(phi)
                        - np.cos(phi) * np.sin(theta) * np.sin(psi)) - vy * (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)) - vx * np.cos(theta) * np.sin(psi), 0,
                        0, 0, np.cos(theta) * np.cos(psi), np.cos(psi) * np.sin(phi) * np.sin(theta) - np.cos(phi) * np.sin(psi),np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta), 0, 0, 0],
                       [-vy * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(theta) * np.sin(psi)) - vz * (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)),
                        vz * np.cos(phi) * np.cos(theta) * np.sin(psi) - vx * np.sin(theta) * np.sin(psi) + vy * np.cos(theta) * np.sin(phi) * np.sin(psi), vz * (np.sin(phi) * np.sin(psi)
                        +np.cos(phi) * np.cos(psi) * np.sin(theta)) - vy * (np.cos(phi) * np.sin(psi) - np.cos(psi) * np.sin(phi) * np.sin(theta)) + vx * np.cos(theta) * np.cos(psi), 0, 0, 0,
                        np.cos(theta) * np.sin(psi), np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi), np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi), 0, 0, 0],
                       [vy * np.cos(phi) * np.cos(theta) - vz * np.cos(theta) * np.sin(phi),- vx * np.cos(theta) - vz * np.cos(phi) * np.sin(theta) - vy * np.sin(phi) * np.sin(theta), 0, 0, 0, 0,
                        -np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(phi) * np.cos(theta), 0, 0, 0]])

    fu = dt * np.array([[0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 1 /Ix, 0., 0.],
                        [0., 0., 1 / Iy, 0.],
                        [0., 0., 0., 1 / Iz],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [-1 / mass, 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.]])
    return fx, fu

def dyn_quad(state,control,dyn):
    phi = state[0]
    theta = state[1]
    psi = state[2]
    phi_rate = state[3]
    theta_rate = state[4]
    psi_rate = state[5]
    vx = state[6]
    vy = state[7]
    vz = state[8]
    g = dyn.g
    mass = dyn.m
    Ix = dyn.Ixx
    Iy = dyn.Iyy
    Iz = dyn.Izz

    ft = control[0]
    tau_x = control[1]
    tau_y = control[2]
    tau_z = control[3]

    f_wx, f_wy, f_wz = 0.0, 0.0, 0.0
    tau_wx, tau_wy, tau_wz = 0.0, 0.0, 0.0

    phi_dot = phi_rate + psi_rate * np.cos(phi) * np.tan(theta) + theta_rate * np.sin(phi) * np.tan(theta)
    theta_dot = theta_rate * np.cos(phi) - psi_rate * np.sin(phi)
    psi_dot = (psi_rate * np.cos(phi)) / np.cos(theta) + (theta_rate * np.sin(phi)) / np.cos(theta)
    phi_rate_dot = (tau_wx + tau_x) / Ix + (theta_rate * psi_rate * (Iy - Iz)) / Ix
    theta_rate_dot = (tau_wy + tau_y) / Iy + (phi_rate * psi_rate * (Ix - Iz)) / Iy
    psi_rate_dot = (tau_wz + tau_z) / Iz + (phi_rate * theta_rate * (Ix - Iy)) / Iz
    vx_dot = psi_rate * vy - theta_rate * vz + f_wx / mass - g * np.sin(theta)
    vy_dot = phi_rate * vz - psi_rate * vx + f_wy / mass + g * np.cos(theta) * np.sin(phi)
    vz_dot = theta_rate * vx - phi_rate * vy - (ft - f_wz + g) / mass + g * np.cos(phi) * np.cos(
        theta)
    x_dot = vz * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) - vy * (
                np.cos(phi) * np.sin(psi) - np.cos(psi) * np.sin(phi) * np.sin(theta)) + vx * np.cos(theta) * np.cos(
        psi)
    y_dot = vy * (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)) - vz * (
                np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(theta) * np.sin(psi)) + vx * np.cos(theta) * np.sin(
        psi)
    z_dot = vz * np.cos(phi) * np.cos(theta) - vx * np.sin(theta) + vy * np.cos(theta) * np.sin(phi)

    f = np.array(
        [phi_dot, theta_dot, psi_dot, phi_rate_dot, theta_rate_dot, psi_rate_dot, vx_dot, vy_dot,
         vz_dot, x_dot, y_dot, z_dot])*dyn.dt
    return f + state
    

def plot_target_and_initial_pnt(x, y, z,par_ddp,ax1,sz):
    ax1.scatter3D(par_ddp.xf[9],par_ddp.xf[10],par_ddp.xf[11],c = "red",s = 2*sz)
    ax1.scatter3D(x[0], y[0], z[0],c = "blue",s = sz)
    return 0
        
def plot_trj(x,y,z,par_ddp,ax1,sz,line_color,line_width):
    ax1.plot3D(x,y,z,color = line_color, linewidth = line_width)
    ax1.scatter3D(par_ddp.xf[9],par_ddp.xf[10],par_ddp.xf[11],c = "red",s = 2*sz)
    ax1.scatter3D(x[0], y[0], z[0],c = "blue",s = sz)
    ax1.scatter3D(x[-1], y[-1],z[-1], c = "green",s = sz)    
    #ax1.quiver(x[0], y[0], math.cos(theta[0]), math.sin(theta[0]), scale=30)
    #ax1.quiver(x[-1], y[-1], math.cos(theta[-1]), math.sin(theta[-1]),scale=30,color = "green")
    return 0
    
def plot_obs(options_lagr,ax1):
    rad_con = options_lagr.rad_con
    center_con = options_lagr.center_con
    num_con = options_lagr.num_con
    N=50
    stride=2
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    for i in range(num_con):
        #circle = plt.Circle(center_con[i,:],rad_con[i], color = 'gray')
        #ax1.add_patch(circle)
        r = rad_con[i]
        cc = center_con[i,:]
        ax1.plot_surface(r*x + cc[0], r*y + cc[1], r*z + cc[2],
                        linewidth=0.0, cstride=stride, rstride=stride,color='gray')
    return 0
    



def graph_quad(x_ddp,u_ddp,par_dyn,par_ddp,options_lagr):
    N = par_ddp.N
    sz = 40
    x = x_ddp[9,:]
    y = x_ddp[10,:]
    z = x_ddp[11,:]
    #theta = x_ddp[2,:]
    fig = plt.figure()
    ax1 = plt.axes(projection ="3d")
    plot_target_and_initial_pnt(x, y, z,par_ddp, ax1,sz)  
    if options_lagr:
        plot_obs(options_lagr, ax1)
    plot_trj(x, y, z, par_ddp, ax1, sz,'blue',2)
    #ax1.set_aspect('equal', 'box')
    utime = par_dyn.dt*np.linspace(0,N-2,N-1)#0 to N-2
    plt.figure()
    plt.plot(utime,u_ddp[0,:],c = "blue")
    plt.plot(utime,u_ddp[1,:],c = "red")
    plt.plot(utime,u_ddp[2,:],c = "green")
    plt.plot(utime,u_ddp[3,:],c = "purple")
    plt.xlabel("Time[s]")
    plt.ylabel("control[N]")
    return 0

def graph_quad_with_noise(x_ddp,u_ddp,x_noise,par_dyn,par_ddp,options_lagr,cnf_reg_flag):
    N = par_ddp.N
    sz = 40
    x_ref = x_ddp[9,:]
    y_ref = x_ddp[10,:]
    z_ref = x_ddp[11,:]
    fig = plt.subplots()
    ax1 = plt.axes(projection ="3d")
    
    plot_target_and_initial_pnt(x_ref, y_ref, z_ref,par_ddp, ax1,sz)  
    
    #confidence
    u = np.linspace(0.0, 2.0 * np.pi, 50)
    v = np.linspace(0.0, np.pi, 50)
    xx_org = np.outer(np.cos(u), np.sin(v))
    yy_org = np.outer(np.sin(u), np.sin(v))
    zz_org = np.outer(np.ones_like(u), np.cos(v))
    stride=2
    STD = 2 #scaling paramter of normal standard div
    conf = 2*norm.cdf(STD)-1 #double tail
    scale = chi2.ppf(conf, df=3)
    #averaging ove num_of_trj to get sequence of mu
    Mu = np.mean(x_noise[0:3,:,:], axis = 2)
    #get state trajectory with noise at each time step
    if cnf_reg_flag:
        for k in range(N):
            xx = xx_org.copy()
            yy = yy_org.copy()
            zz = zz_org.copy()
        
            x_k = x_noise[0:3,k,:].copy() #nx(only pos) by num_trj
            Mu_k = Mu[0:3,k]
            cov = np.asarray(np.cov(x_k))
            D,V = np.linalg.eig(cov*scale);# d eigenvalue, v is eigen matrix
            order = np.flip(np.argsort(D)) #large to small
            D = np.diag(D)
            V = V[:,order]
            sqrt_D = np.sqrt(D)
            VV = V@sqrt_D
            for i in range(len(xx)):
                for j in range(len(xx)):
                    [xx[i,j],yy[i,j],zz[i,j]] = VV@[xx[i,j],yy[i,j],zz[i,j]]+Mu_k
                    ax1.plot_surface(xx, yy, zz,linewidth=0.0, cstride=stride, rstride=stride,color='purple', alpha=0.1)
    if options_lagr:
        plot_obs(options_lagr, ax1)
    num_trj = x_noise.shape[2]
    for i in range(num_trj):
        x = x_noise[0,:,i].copy()
        y = x_noise[1,:,i].copy()
        z = x_noise[2,:,i].copy()
        #plot_trj(x, y, z, par_ddp, ax1, sz,'red',0.5)
    plot_trj(x_ref, y_ref, z_ref, par_ddp, ax1, sz,'blue',2)
    
    

    

    
    
    
    #ax1.set_aspect('equal', 'box')
    utime = par_dyn.dt*np.linspace(0,N-2,N-1)#0 to N-2
    plt.figure()
    plt.plot(utime,u_ddp[0,:],c = "blue")
    plt.plot(utime,u_ddp[1,:],c = "red")
    plt.plot(utime,u_ddp[2,:],c = "green")
    plt.plot(utime,u_ddp[3,:],c = "purple")
    plt.xlabel("Time[s]")
    plt.ylabel("control[N]")
    return 0
    
    