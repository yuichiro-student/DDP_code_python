# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:22:44 2022

@author: yaoya
"""

import numpy as np


class Quadrotor:
    def __init__(self, dt):

        
        # pysical parameters are coming from this paper
        # https://sal.aalto.fi/publications/pdf-files/eluu11_public.pdf
        self.n_x = 12                #number of states
        self.n_u = 4                 #number of controls
        self.dt = dt
        self.g = 9.81                # gravity (m/s^2)
        self.m  = 0.468;             #[kg]     mass of quadrotor
        self.Ixx  = 4.856*10**(-3);   #[kgs^2]    inertia of quadrotor
        self.Iyy  = 4.856*10**(-3);   #[kgs^2]    inertia of quadrotor
        self.Izz  = 8.801*10**(-3);   #[kgs^2]    inertia of quadrotor
        self.l = 0.225;              #[m]arm length of rotor;
        self.k = 2.980*10**(-6);     #lift constant for the rotor 
        self.b = 1.140*10**(-7);     #drag constant for the rotor
        self.kd  = 0.25;             #drag constant for the quad
        
        
        
        #self.x0 = np.array([[4], [4], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
        #self.xf = np.array([[-4], [-2], [-1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
        
        self.x0 = np.array([4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.xf = np.array([-4, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def system_dyn(self, state, control):
        m = self.m
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz;
        l = self.l;
        g = self.g;
        kd = self.kd;
        x = state
        #x[0]-x[2]: Position of COG in World Frame.
        #x[3]-x[5]: Velocity of COG in Body Frame.
        #x[6]-x[8]: Rotation of the quad. in WF.
        #x[9]-x[11]: Angluar vel of the quad in BF.
        u = control     #lift force of rotors
        #tau: moments caused by rotors
        #F: total lif force of rotors
        tau1 = l*(u[3]-u[1]);    tau2 = l*(u[2]-u[0]);
        tau3 = self.b/self.k*(u[1]-u[0]-u[2]+u[3]);
        F = u[1]+u[2]+u[3]+u[0];
        
        #recover angles from coefficients and trig. functions
        phi = x[6]; theta = x[7]; psi = x[8]; 
        p = x[9]; q = x[10]; r = x[11];
        cphi = np.cos(phi); sphi = np.sin(phi); cth = np.cos(theta);  sth = np.sin(theta);
        cpsi = np.cos(psi); spsi = np.sin(psi);
        tth = np.tan(theta);

        #third column of rotation matrix times vertical force
        RotF = np.array([cpsi*sth*cphi+spsi*sphi, spsi*sth*cphi-cpsi*sphi, cth*cphi])*F
        #angular velocity in WF
        Winv_omega = np.array([p+sphi*tth*q+cphi*tth*r, cphi*q-sphi*r, sphi*q/cth + cphi*r/cth]);

        a = x[3:6]; 
        B = np.array([0, 0, -g]) + RotF/m -kd/m*x[3:6];
        C = Winv_omega;
        D = np.array([(Iyy-Izz)*q*r/Ixx, (Izz-Ixx)*p*r/Iyy, (Ixx-Iyy)*p*q/Izz])+np.array([tau1/Ixx, tau2/Iyy, tau3/Izz])

        new_state = np.concatenate((a,B,C,D), axis=None)
        return new_state

    def system_propagate(self, state, control):
        f = self.system_dyn(state, control)
        state_next = state + self.dt * f
        return state_next

    def system_grad(self, state, control):
        dt = self.dt
        m = self.m
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz;
        l = self.l;
        kd = self.kd;
        x = state
        #x[0]-x[2]: Position of COG in World Frame.
        #x[3]-x[5]: Velocity of COG in Body Frame.
        #x[6]-x[8]: Rotation of the quad. in WF.
        #x[9]-x[11]: Angluar vel of the quad in BF.
        u = control     #lift force of rotors
        #tau: moments caused by rotors
        #F: total lif force of rotors
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
        Winv = np.array([[1, sphi*tth, cphi*tth],[0, cphi, -sphi],[0, sphi/cth, cphi/cth]]);
        
        a1 = (Iyy-Izz)/Ixx; a2 = (Izz-Ixx)/Iyy; a3 = (Ixx-Iyy)/Izz;
        omega_domega = np.array([[0, a1*r, a1*q], [a2*r, 0, a2*p],[a3*q, a3*p, 0]]);


        pos_grad = np.concatenate((np.zeros((3,3)), np.identity(3), np.zeros((3,6))), axis=1);
        vel_grad =  np.concatenate((np.zeros((3,3)), -kd/m*np.identity(3), Rot_dangle*F/m, np.zeros((3,3))),axis = 1);
        angle_grad = np.concatenate((np.zeros((3,6)), Winv_angle, Winv), axis = 1);
        omega_grad = np.concatenate((np.zeros((3,9)),omega_domega),axis = 1);
        
        fx = np.identity(self.n_x) + np.concatenate((pos_grad, vel_grad, angle_grad, omega_grad),axis = 0)*dt;


        pos_grad = np.zeros((3,4));
        vel_grad_tmp = np.array([[cpsi*sth*cphi + spsi*sphi], [spsi*sth*cphi - cpsi*sphi],[cth*cphi]])/m #3 by 1
        vel_grad = np.tile(vel_grad_tmp,(1,4)); # 3 by 4
        #Third column of rotation matrix
        angle_grad = np.zeros((3,4));
        omega_grad1 = l/Ixx *np.array([[0, -1, 0, 1]]);
        omega_grad2 = l/Iyy *np.array([[-1, 0, 1, 0]]);
        omega_grad3 = 1/Izz*self.b/self.k*np.array([[-1, 1, -1, 1]]); 
        omega_grad = np.concatenate((omega_grad1, omega_grad2, omega_grad3), axis = 0) #3 by 4
                                          
        fu = np.concatenate((pos_grad,vel_grad,angle_grad,omega_grad),axis = 0)*dt;
        return fx, fu
    