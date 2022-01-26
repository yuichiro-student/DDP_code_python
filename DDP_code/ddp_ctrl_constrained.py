# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:38:29 2022

@author: yaoya
"""

class par_ddp:
    def __init__(self, f_dyn, grad_dyn, L_cost, F_cost, x0, N, xf):
        self.f_dyn =f_dyn
        self.grad_dyn = grad_dyn
        self.L_cost = L_cost
        self.F_cost = F_cost
        self.x0 = x0
        self.N = N
        self.xf = xf


def ddp_ctrl_constrained(u_bar,par_ddp,par_dyn,options_ddp):
    import numpy as np
    import warnings
    from boxQP import boxQP
# first order discrete ddp without and without control constraints
# returns x, u, S, u_bars, x_bars, J, norm_costgrad
#tensdot = @(a,b) permute(sum(bsxfun(@times,a,b),1), [2 3 1]);

#get nominal sequence
    x_k = par_ddp.x0
    n_x = par_dyn.n_x
    n_u = par_dyn.n_u
    N = par_ddp.N
    iter = options_ddp['ddp_iter']
    n_X = len(x_k);
    x = np.zeros((n_X,N));
    x[:,1] = x_k;
    c_L = 0;
    for k in range(N-1):
        u_k = u_bar[:,k].copy();
        c_L = c_L + par_ddp.L_cost(x_k, u_k,k,False);
        x_k = par_ddp.f_dyn(x_k, u_k);
        x[:,k + 1] = x_k;
        
    x_final = x[:,-1].copy()
    c_F = par_ddp.F_cost(x_final, False)
    cost_traj0 = c_L + c_F;  #total cost of nominal sequence
    x_bar = x.copy();
    J = np.zeros(iter);
    u_bars = np.zeros((n_u, N-1,iter));
    x_bars = np.zeros((n_X, N,iter));
    lims = par_dyn.lims
    dV = np.array([0, 0]);
    lambda_reg = options_ddp['lambda_reg']
    dlambda_reg = options_ddp['dlambda_reg']
    lambdaFactor = options_ddp['lambdaFactor']
    lambdaMax = options_ddp['lambdaMax']
    lambdaMin = options_ddp['lambdaMin']

#ddp
    for j in range(iter): #starting from 0 to iter-1
        J[j] = cost_traj0
        u_bars[:,:,j] = u_bar.copy()
        x_bars[:,:,j] = x_bar.copy()
        print('iteration %d out of %d ---- total cost is %f' % (j,iter-1,J[j]));
    
    
    #backward
        backPassDone   = 0;
        while not backPassDone:
            V_x_kplus1, V_xx_kplus1 = par_ddp.F_cost(x_final, True);   
            k_out = np.zeros((n_u,N-1));
            K_out = np.zeros((n_u, n_X, N-1));
        
        
            for k in range(N-2, -1,-1):#N-2 to 0
                x_bar_k = x_bar[:,k].copy();
                u_bar_k = u_bar[:,k].copy();
                #have them as a colum vector
                #x_bar_k = np.expand_dims(x_bar_k,axis=0)
                #u_bar_k = np.expand_dims(x_bar_k,axis=0)
            
                Fi, B = par_ddp.grad_dyn(x_bar_k, u_bar_k);
                L_x, L_u, L_xu, L_uu, L_xx = par_ddp.L_cost(x_bar_k, u_bar_k, k,True);
        
                Qx = L_x + np.matmul(Fi.T,V_x_kplus1) #n_x array
                Qu = L_u + np.matmul(B.T,V_x_kplus1)  #n_u array
                Qxx = L_xx + np.matmul(Fi.T, np.matmul(V_xx_kplus1,Fi))
                Qxu = L_xu + np.matmul(Fi.T, np.matmul(V_xx_kplus1,B))
                Quu = L_uu + np.matmul(B.T, np.matmul(V_xx_kplus1,B))
            
            
                #if isfield(par_ddp, 'hess_dyn')
                #    [fxx,fxu,fuu] = par_ddp.hess_dyn(x_bar(:,k), u_bar(:,k));
                #    Qxx=Qxx+tensdot(V_x_kplus1,fxx);
                #    Qxu=Qxu+tensdot(V_x_kplus1,fxu);
                #    Quu = Quu + tensdot(V_x_kplus1,fuu);
                #end
            
                Qux = Qxu.T
                Quu_reg = Quu + lambda_reg*np.identity(n_u);
            
                #if any(isnan(Quu_reg))
                #    disp('lll')
                #end
                if lambda_reg>=lambdaMax:  
                    print('Regularizer is too large');
                    backPassDone = False;
                    break
            
                eig, tmp = np.linalg.eig(Quu_reg)
                min_eig = min(eig)
                if any(eig <= 0):
                    dlambda_reg   = max(dlambda_reg * lambdaFactor, lambdaFactor);
                    lambda_reg    = max(lambda_reg * dlambda_reg, lambdaMin);
                    print('Quu ndf at %d with minimun eig.: %6f lambda: %6f' % (k,min_eig, lambda_reg));
                    break            
                if not lims.size == 0: 
                    #box constraint QP
                    upper = lims[0,:] - u_bar_k;
                    lower = lims[1,:] - u_bar_k;
                    
                    x0_guess = k_out[:,min(k+1,N-2)]
                    k_k,result,Hf,free,trace = boxQP(Quu_reg, Qu,lower, upper, x0_guess)
                    #l_ans,result,Hf,free = boxQP(Quu_reg,Qu,lower,upper,l(:,min(k+1,N-2))); 
                    K_k    = np.zeros((n_u, n_X))
                    if any(free) and result > 1:
                        #Lfree        =  -Hf\(Hf'\Qux(free,:));
                        Lfree        =  -np.linalg.solve(Hf,np.linalg.solve(Hf.T, Qux[free,:]))
                        K_k[free,:] = Lfree;
                else:
                   k_k = -np.linalg.solve(Quu_reg, Qu) #Quu_reg^(-1)*Qu, n_u array
                   K_k = -np.linalg.solve(Quu_reg, Qux)  #Quu_reg^(-1)*Qux
               
           
                K_out[:,:,k] =  K_k
                k_out[:,k] = k_k
                
                dV = dV + np.array([np.dot(k_k,Qu), 0.5*np.dot(k_k,Quu@k_k)]);
                #dV = dV + [l[:,k]'*Qu  .5*l(:,k)'*Quu*l(:,k)];
                #V_x_kplus1 = Qx + S(:,:,k)'*Quu*l(:,k) + S(:,:,k)'*Qu + Qxu*l(:,k); %Qx + Qxu*l(:,k);
                V_x_kplus1 = Qx + K_k.T@Quu@k_k + K_k.T@Qu + Qxu@k_k #Qx + Qxu*l(:,k);
                #V_xx_kplus1 = Qxx + S(:,:,k)'*Quu*S(:,:,k) + S(:,:,k)'*Qux + Qxu*S(:,:,k); %Qxx + Qxu*S(:,:,k);
                #V_xx_kplus1 = 0.5 * (V_xx_kplus1 + V_xx_kplus1');
                V_xx_kplus1 = Qxx + K_k.T@Quu@K_k + K_k.T@Qux + Qxu@K_k# %Qxx + Qxu*S(:,:,k);
                V_xx_kplus1 = 0.5 * (V_xx_kplus1 + V_xx_kplus1.T)
            
                if k==0:
                    backPassDone = 1;
    
    #check for termination due to small gradient
        tmp = (abs(k_out) / (abs(u_bar)+1))
        g_norm = np.mean(tmp.max(0))
        #g_norm         = mean(max(abs(l) / (abs(u_bar)+1),[],1));
        if g_norm < 1e-4 and lambda_reg < 1e-5:
            dlambda_reg  = min(dlambda_reg / lambdaFactor, 1/lambdaFactor);
            lambda_reg    = lambda_reg * dlambda_reg * (lambda_reg > lambdaMin);
            print('\nSUCCESS: gradient norm < tolGrad\n');
            norm_costgrad = 0;
            x = x_bar; u = u_bar;
            break
     
    #forward
        fwdPassDone = False;
        if backPassDone:
            x = np.zeros((n_X,N));
            alpha0 = 10**np.linspace(0,-3,11);
            for alpha in alpha0: 
                    x_k = par_ddp.x0;
                    x[:,1] = x_k;
                    deltax_k = np.zeros(n_X);
                    u = np.zeros((n_u,N - 1));
                    c_L = 0;
                    for k in range(N-1):#0 to N-2
                        deltau_k = alpha*k_out[:,k] + np.matmul(K_out[:,:,k],deltax_k)
                        u_k = u_bar[:,k] + deltau_k;
                        if not lims.size == 0:
                            u_k = np.minimum(lims[0,:], np.maximum(lims[1,:], u_k));
                        
                        c_L = c_L + par_ddp.L_cost(x_k, u_k, k,False);
                        u[:,k] = u_k;
                        x_k = par_ddp.f_dyn(x_k, u_k);
                                
                        if np.isinf(x).any() or np.isinf(c_L) or np.isnan(x).any():
                            print('INF state\n');
                            break            
                        x[:,k + 1] = x_k;
                        deltax_k = x[:,k + 1] - x_bar[:,k + 1];        
            
                    x_final = x[:,-1].copy();
                    c_F = par_ddp.F_cost(x_final, False);
                    cost_traj1 = c_L + c_F;
                    dcost = cost_traj0-cost_traj1;
                    expected = -alpha*(dV[0] + alpha*dV[1]);
                    if expected > 0:
                        z = dcost/expected;
                    else:
                        z = np.sign(dcost);
                        warnings.warn("Warning...........Message\n")
            
                    if (z > 0):
                        fwdPassDone = 1
                        break
 #accept or reject
        if fwdPassDone:
        #decrease lambda
            dlambda_reg   = min(dlambda_reg / lambdaFactor, 1/lambdaFactor);
            lambda_reg    = lambda_reg * dlambda_reg * (lambda_reg > lambdaMin);        
            conv_flag = 0;
            if dcost < options_ddp['cost_tol']:           
                conv_flag = 1
            
            cost_traj0 = cost_traj1;
            x_bar = x; u_bar = u;
            norm_costgrad = abs(cost_traj1-cost_traj0) / cost_traj0;
            #YUICHIRO
            norm_costgrad = dcost;
            #
            if conv_flag:
                print('convergence of DDP dcost: %.3f\n' % (dcost));
                break
        else:
            dlambda_reg   = max(dlambda_reg * lambdaFactor, lambdaFactor);
            lambda_reg    = max(lambda_reg * dlambda_reg, lambdaMin);
            print('Not enough cost reduction make lambda larger,lambda:%2f\n' % (lambda_reg))
            if lambda_reg > lambdaMax:
                norm_costgrad = 0;
                x = x_bar;
                u = u_bar;
                print('\nEXIT: lambda > lambdaMax\n');
                break
    J = J[0:j];
    u_bars = u_bars[:, :, 0:j-1];
    x_bars = x_bars[:, :, 0:j-1];
    return x, u, K_out, u_bars, x_bars, J, norm_costgrad

