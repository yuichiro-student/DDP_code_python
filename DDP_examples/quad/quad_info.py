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
    def __init__(self, n_x, n_u, dt, lims,m, Ixx, Iyy, Izz, l, k, b, kd, g):
        self.n_x  = n_x
        self.n_u  = n_u
        self.dt = dt
        self.lims = lims
        self.m = m
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.l = l
        self.k = k
        self.b = b
        self.kd = kd
        self.g = g
        
def grad_dyn_quad(x,u,dyn):
    dt = dyn.dt
    m = dyn.m
    Ixx = dyn.Ixx
    Iyy = dyn.Iyy
    Izz = dyn.Izz;
    l = dyn.l;
    g = dyn.g;
    kd = dyn.kd;

    F = u[1]+u[2]+u[3]+u[0];

    #recover angles from coefficients and trig. functions
    phi = x[6]; theta = x[7]; psi = x[8]; 
    p = x[9]; q = x[10]; r = x[11];
    cphi = np.cos(phi); sphi = np.sin(phi); cth = np.cos(theta);  sth = np.sin(theta);
    cpsi = np.cos(psi); spsi = np.sin(psi);
    tth = np.tan(theta);
    #derivative of third column of rotation matrix 
    Rot_dangle = np.array([[cphi*spsi - cpsi*sphi*sth, cphi*cth*cpsi, cpsi*sphi - cphi*sth*spsi],
                           [-cphi*cpsi - sphi*sth*spsi, cphi*cth*spsi, sphi*spsi + cphi*cpsi*sth],
                           [-cth*sphi,   -cphi*sth,0]]);
    
    #derivative of angle in body frame(Winv*omega) 
    Winv_phi_omega = np.array([[q*cphi*tth-r*sphi*tth], [-q*sphi-r*cphi], [q*cphi/cth-r*sphi/cth]]); #3 by 1
    Winv_tht_omega = np.array([[q*sphi*(tth**2 + 1)+r*cphi*(tth**2 + 1)], [0],
                               [q*(sphi*sth)/(cth**2)+r*(cphi*sth)/(cth**2)]]) # 3 by 1
    Winv_angle = np.concatenate((Winv_phi_omega, Winv_tht_omega, np.zeros((3,1))),axis = 1);

    #Winv_phi_omega = [q.*cphi.*tth-r.*sphi*tth; -q.*sphi-r.*cphi;q.*cphi./cth-r.*sphi./cth];
    #Winv_tht_omega = [q.*sphi*(tth^2 + 1)+r.*cphi*(tth.^2 + 1);0; q.*(sphi.*sth)./(cth.^2)+r.*(cphi.*sth)./(cth.^2)];
    #Winv_angle = [Winv_phi_omega, Winv_tht_omega, zeros(3,1)];                            

                                
    Winv = np.array([[1, sphi*tth, cphi*tth],[0, cphi, -sphi],[0, sphi/cth, cphi/cth]]);
    a1 = (Iyy-Izz)/Ixx; a2 = (Izz-Ixx)/Iyy; a3 = (Ixx-Iyy)/Izz;
    omega_domega = np.array([[0, a1*r, a1*q], [a2*r, 0, a2*p],[a3*q, a3*p, 0]]);


    pos_grad = np.concatenate((np.zeros((3,3)), np.identity(3), np.zeros((3,6))), axis=1);
    vel_grad =  np.concatenate((np.zeros((3,3)), -kd/m*np.identity(3), Rot_dangle*F/m, np.zeros((3,3))),axis = 1);
    angle_grad = np.concatenate((np.zeros((3,6)), Winv_angle, Winv), axis = 1);
    omega_grad = np.concatenate((np.zeros((3,9)),omega_domega),axis = 1);
    
    fx = np.identity(12) + np.concatenate((pos_grad, vel_grad, angle_grad, omega_grad),axis = 0)*dyn.dt;


    pos_grad = np.zeros((3,4));
    vel_grad_tmp = np.array([[cpsi*sth*cphi + spsi*sphi], [spsi*sth*cphi - cpsi*sphi],[cth*cphi]])/m #3 by 1
    vel_grad = np.tile(vel_grad_tmp,(1,4)); # 3 by 4
    #Third column of rotation matrix
    angle_grad = np.zeros((3,4));
    omega_grad1 = l/Ixx *np.array([[0, -1, 0, 1]]);
    omega_grad2 = l/Iyy *np.array([[-1, 0, 1, 0]]);
    omega_grad3 = 1/Izz*dyn.b/dyn.k*np.array([[-1, 1, -1, 1]]); 
    omega_grad = np.concatenate((omega_grad1, omega_grad2, omega_grad3), axis = 0) #3 by 4
                                      
    fu = np.concatenate((pos_grad,vel_grad,angle_grad,omega_grad),axis = 0)*dt;
    return fx, fu

def dyn_quad(x,u,dyn):
    dt = dyn.dt
    m = dyn.m
    Ixx = dyn.Ixx
    Iyy = dyn.Iyy
    Izz = dyn.Izz;
    l = dyn.l;
    g = dyn.g;
    kd = dyn.kd;

    tau1 = l*(u[3]-u[1]);    tau2 = l*(u[2]-u[0]);
    F = u[1]+u[2]+u[3]+u[0];
    tau3 = dyn.b/dyn.k*(u[1]-u[0]-u[2]+u[3]);

    #recover angles from coefficients and trig. functions
    phi = x[6]; theta = x[7]; psi = x[8]; 
    p = x[9]; q = x[10]; r = x[11];
    cphi = np.cos(phi); sphi = np.sin(phi); cth = np.cos(theta);  sth = np.sin(theta);
    cpsi = np.cos(psi); spsi = np.sin(psi);
    tth = np.tan(theta);

    #third column of rotation matrix times vertical force
    RotF = np.array([cpsi*sth*cphi+spsi*sphi, spsi*sth*cphi-cpsi*sphi, cth*cphi])*F
    #angular velocity in inertia (fixed, reference) frame
    Winv_omega = np.array([p+sphi*tth*q+cphi*tth*r, cphi*q-sphi*r, sphi*q/cth + cphi*r/cth]);

    a = x[3:6];
    B = np.array([0, 0, -g]) + RotF/m -kd/m*x[3:6];#velocity
    C = Winv_omega;
    D = np.array([(Iyy-Izz)*q*r/Ixx, (Izz-Ixx)*p*r/Iyy, (Ixx-Iyy)*p*q/Izz])+np.array([tau1/Ixx, tau2/Iyy, tau3/Izz])

    x_new = x + np.concatenate((a,B,C,D), axis=None)*dt;
    return x_new
    

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
    
    