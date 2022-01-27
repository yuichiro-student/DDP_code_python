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

    dt = dyn.dt
    m = dyn.m
    Ixx = dyn.Ixx
    Iyy = dyn.Iyy
    Izz = dyn.Izz
    l = dyn.l
    phi, theta, psi = state.item(0), state.item(1), state.item(2)  # roll, pitch and yaw angles in earth frame
    phi_rate, theta_rate, psi_rate = state.item(3), state.item(4), state.item(
        5)  # roll, pitch, yaw velocities in body frame
    vx, vy, vz = state.item(6), state.item(7), state.item(8)  # x, y, and z linear velocities in body frame
    ft = control.item(0)  # thrust in body frame
    tau_x, tau_y, tau_z = control.item(1), control.item(2), control.item(3)  # x,y and z torques in the body frame

    fx = np.eye(self.n) + self.dt * np.array([
        [vy * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) + vz * (np.cos(phi) * np.sin(psi) - np.cos(psi) * np.sin(phi) * np.sin(theta)),
         vz * np.cos(phi) * np.cos(theta) * np.cos(psi) - vx * np.cos(psi) * np.sin(theta) + vy * np.cos(theta) * np.cos(psi) * np.sin(phi),
         vz * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(theta) * np.sin(psi)) - vy * (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)) - vx * np.cos(theta) * np.sin(psi),
         0, 0, 0, np.cos(theta) * np.cos(psi), np.cos(psi) * np.sin(phi) * np.sin(theta) - np.cos(phi) * np.sin(psi), np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta), 0, 0, 0],
        [-vy * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(theta) * np.sin(psi)) - vz * (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)),
         vz * np.cos(phi) * np.cos(theta) * np.sin(psi) - vx * np.sin(theta) * np.sin(psi) + vy * np.cos(theta) * np.sin(phi) * np.sin(psi), vz * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) - vy * (
                 np.cos(phi) * np.sin(psi) - np.cos(psi) * np.sin(phi) * np.sin(theta)) + vx * np.cos(theta) * np.cos(psi), 0, 0,0, np.cos(theta) * np.sin(psi),np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi),
                np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi), 0, 0, 0],
        [vy * np.cos(phi) * np.cos(theta) - vz * np.cos(theta) * np.sin(phi),
         - vx * np.cos(theta) - vz * np.cos(phi) * np.sin(theta) - vy * np.sin(
             phi) * np.sin(theta), 0, 0, 0, 0, -np.sin(theta),
         np.cos(theta) * np.sin(phi), np.cos(phi) * np.cos(theta), 0, 0, 0],
        [0, -self.g * np.cos(theta), 0, 0, -vz, vy, 0, psi_rate, -theta_rate, 0,
         0, 0],
        [self.g * np.cos(phi) * np.cos(theta),
         -self.g * np.sin(phi) * np.sin(theta), 0, vz, 0, -vx, -psi_rate, 0,
         phi_rate, 0, 0, 0],
        [-self.g * np.cos(theta) * np.sin(phi),
         -self.g * np.cos(phi) * np.sin(theta), 0, -vy, vx, 0, theta_rate,
         -phi_rate, 0, 0, 0, 0],
        [theta_rate * np.cos(phi) * np.tan(theta) - psi_rate * np.sin(phi) * np.tan(theta),
         psi_rate * np.cos(phi) * (np.tan(theta) ** 2 + 1) + theta_rate * np.sin(phi) * (
                                                           np.tan(theta) ** 2 + 1), 0, 1, np.sin(phi) * np.tan(theta),
                                               np.cos(phi) * np.tan(theta), 0, 0, 0, 0, 0, 0],
                                              [-psi_rate * np.cos(phi) - theta_rate * np.sin(phi), 0, 0, 0, np.cos(phi),
                                               -np.sin(phi), 0, 0, 0, 0, 0, 0],
                                              [(theta_rate * np.cos(phi)) / np.cos(theta) - (
                                                          psi_rate * np.sin(phi)) / np.cos(theta),
                                               (psi_rate * np.cos(phi) * np.sin(theta)) / np.cos(theta) ** 2 + (
                                                           theta_rate * np.sin(phi) * np.sin(theta)) / np.cos(
                                                   theta) ** 2, 0, 0, np.sin(phi) / np.cos(theta),
                                               np.cos(phi) / np.cos(theta), 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, (psi_rate * (self.Iy - self.Iz)) / self.Ix,
                                               (theta_rate * (Iyy - Izz)) / self.Ix, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, -(psi_rate * (Ixx - Izz)) / Iyy, 0,
                                               -(phi_rate * (Ixx - Izz)) / Iyy, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, (theta_rate * (Ixx - Iyy)) / self.Iz,
                                               (phi_rate * (Ixx - Iyy)) / Izz, 0, 0, 0, 0, 0, 0, 0]
                                              ])

    fu = self.dt * np.array([[0., 0., 0., 0.],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.],
                            [0., 0., 0., 0.],
                            [0., 0., 0., 0.],
                            [-1 / self.mass, 0., 0., 0.]
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.],
                             [0., 1 / self.Ix, 0., 0.],
                             [0., 0., 1 / self.Iy, 0.],
                             [0., 0., 0., 1 / self.Iz]])
    return fx, fu

def dyn_quad(state,control,dyn):
    phi = state[6]
    theta = state[7]
    psi = state[8]
    p = state[9]
    q = state[10]
    r = state[11]
    u = state[3]
    v = state[4]
    w = state[5]
    x = state[0]
    y = state[1]
    z = state[2]

    tau_x = control[0]
    tau_y = control[1]
    tau_z = control[2]
    f_t = control[3] + 9.81

    f_wx, f_wy, f_wz = 0.0, 0.0, 0.0
    tau_wx, tau_wy, tau_wz = 0.0, 0.0, 0.0

    return state + np.array([
        w * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)) - v * (
                cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta)) + u * (cos(psi) * cos(theta)),
        v * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta)) - w * (
                cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)) + u * (cos(theta) * sin(psi)),
        w * cos(phi) * cos(theta) - u * sin(theta) + v * cos(theta) * sin(phi),
        r * v - q * w - g * sin(theta) + f_wx / m,
        p * w - r * u + g * sin(phi) * cos(theta) + f_wy / m,
        q * u - p * v + g * cos(theta) * cos(phi) + (f_wz - f_t) / m,
        p + r * cos(phi) * tan(theta) + q * sin(phi) * tan(theta),
        q * cos(phi) - r * sin(phi),
        r * cos(phi) / cos(theta) + q * sin(phi) / cos(theta),
        (Iy - Iz) / Ix * r * q + (tau_x + tau_wx) / Ix,
        (Iz - Ix) / Iy * p * r + (tau_y + tau_wy) / Iy,
        (Ix - Iy) / Iz * p * q + (tau_z + tau_wz) / Iz,
    ])*dyn.dt
    

def plot_target_and_initial_pnt(x,y,z,par_ddp,ax1,sz):
    ax1.scatter3D(par_ddp.xf[0],par_ddp.xf[1],par_ddp.xf[2],c = "red",s = 2*sz)
    ax1.scatter3D(x[0], y[0], z[0],c = "blue",s = sz)
    return 0
        
def plot_trj(x,y,z,par_ddp,ax1,sz,line_color,line_width):
    ax1.plot3D(x,y,z,color = line_color, linewidth = line_width)
    ax1.scatter3D(par_ddp.xf[0],par_ddp.xf[1],par_ddp.xf[2],c = "red",s = 2*sz)
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
    x = x_ddp[0,:]
    y = x_ddp[1,:]
    z = x_ddp[2,:]
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
    x_ref = x_ddp[0,:]
    y_ref = x_ddp[1,:]
    z_ref = x_ddp[2,:]
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
    
    