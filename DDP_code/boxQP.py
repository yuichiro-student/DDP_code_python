# -*- coding: utf-8 -*-
def boxQP(H,g,lower,upper,x0,trace_out=False,verbose=False):
    import numpy as np
    from scipy.linalg import cholesky
# Minimize 0.5*x'*H*x + x'*g  s.t. lower<=x<=upper
#
#  inputs:
#     H            - positive definite matrix   (n * n)
#     g            - bias vector                (n)
#     lower        - lower bounds               (n)
#     upper        - upper bounds               (n)
#
#   optional inputs:
#     x0           - initial state              (n)
#     options      - see below                  (7)
#
#  outputs:
#     x            - solution                   (n)
#     result       - result type (roughly, higher is better, see below)
#     Hfree        - subspace cholesky factor   (n_free * n_free)
#     free         - set of free dimensions     (n)
    #if nargin==0:
    #    demoQP(); # run the built-in demo
   #return;

    n        = H.shape[0]
    clamped  = np.zeros(n, dtype=bool)
    free     = np.ones(n, dtype=bool)
    oldvalue = 0
    result   = 0
    gnorm    = 0
    nfactor  = 0
    trace    = [];
    Hfree    = np.zeros((n,n))
    clamp    = lambda x: np.maximum(lower, np.minimum(upper, x))
    
    x = clamp(x0)       # initial state
    x[np.isinf(x)] = 0  # set inf state to zero
    # options
    #if nargin > 5
    #    options        = num2cell(options(:));
    #    [maxIter, minGrad, minRelImprove, stepDec, minStep, Armijo, print] = deal(options{:});
    #else % defaults
    maxIter        = 100       # maximum number of iterations
    minGrad        = 1e-8      # minimum norm of non-fixed gradient
    minRelImprove  = 1e-8      # minimum relative improvement
    stepDec        = 0.6       # factor for decreasing stepsize
    minStep        = 1e-22     # minimal stepsize for linesearch
    Armijo         = 0.1   	# Armijo parameter (fraction of linear improvement required)
	#print          = 0;	    #verbosity
#end
# initial objective value
    value    = np.dot(x,g) + 0.5*np.dot(x,np.matmul(H, x))    
    #print(value)
    if verbose:
       print("==========\nStarting box-QP, dimension %3d, initial value: %12.3f\n" % (n, value))

# main loop
    for iter in range(maxIter):
        if not (result == 0):
            break
    
        # check relative improvement
        if iter>0 and (oldvalue - value) < minRelImprove*abs(oldvalue):
            result = 4
            break
    
        oldvalue = value;
        #get gradient
        grad     = g + np.matmul(H,x);
    
        # find clamped dimensions
        old_clamped                     = clamped;
        clamped                         = np.zeros(n, dtype=bool);
        clamped[np.logical_and((x == lower), (grad>0))] = True;
        clamped[np.logical_and((x == upper), (grad<0))] =  True;
        free                            = np.invert(clamped);
    
        #check for all clamped
        if all(clamped):
            result = 6;
            break;
    
        # factorize if clamped has changed
        if iter == 0:
            factorize    = True;
        else:
            #factorize    = any(old_clamped ~= clamped);
            factorize    = any(old_clamped != clamped);
    
    
        if factorize:
        #[Hfree, indef]  = chol(H(free,free));
        #H_free = H[free,free]
            tmp = H[free]
            H_free = tmp[:,free]
            eig, tmp = np.linalg.eig(H_free)
        
        if any(eig <= 0):
            result = -1;
            break
        else:
            Hfree = cholesky(H_free, lower=False)
        
        nfactor            = nfactor + 1;
    
    
        #check gradient norm
        gnorm  = np.linalg.norm(grad[free]);
        if gnorm < minGrad:
            result = 5;
            break

    
        # get search direction
        grad_clamped   = g  + np.matmul(H,np.multiply(x,clamped))
        search         = np.zeros(n)
        search[free]   = -1*np.linalg.solve(Hfree,np.linalg.solve(Hfree.T,grad_clamped[free])) - x[free];
        #np.linalg.solve(A, B)#A^{-1}B
                         
                                 
        #check for descent direction
        sdotg          = np.dot(search,grad)
        if sdotg >= 0:# (should not happen)
            break


        #armijo linesearch
        step  = 1;
        nstep = 0;
        xc    = clamp(x+step*search)
        vc    = np.matmul(xc.T,g) + 0.5*np.dot(xc,np.matmul(H,xc))
        while (vc - oldvalue)/(step*sdotg) < Armijo:
            step  = step*stepDec;
            nstep = nstep+1;
            xc = clamp(x+step*search);
            vc    = np.dot(xc,g) + 0.5*np.dot(xc,np.matmul(H,xc));
            if step<minStep:
                result = 2;
                break
        
        if verbose:
            print("iter %3d  value %9.5g |g| %9.3g  reduction %9.3g  linesearch %g^%2d  n_clamped %d\n" % (iter, vc, gnorm, oldvalue-vc, stepDec, nstep, sum(clamped)))
    
        if trace_out:
            #trace[iter].x        = x; #ok<*AGROW>
            #trace[iter].xc       = xc;
            #trace[iter].value    = value;
            #trace[iter].search   = search;
            #trace[iter].clamped  = clamped;
            #trace[iter].nfactor  = nfactor;
            trace.append([x,xc,value,clamped,nfactor])
    
        #accept candidate
        x     = xc;
        value = vc;

    if iter >= maxIter:
        result = 1;

    results = ["Hessian is not positive definite",  # result = -1
            "No descent direction found",                # result = 0    SHOULD NOT OCCUR
            "Maximum main iterations exceeded",          # result = 1
            'Maximum line-search iterations exceeded',   # result = 2
            'No bounds, returning Newton point',         # result = 3
            'Improvement smaller than tolerance',        # result = 4
            'Gradient norm smaller than tolerance',      # result = 5
            'All dimensions are clamped']                  # result = 6
    if verbose:
        print('RESULT: %s.\niterations %d  gradient %-12.6g final value %-12.6g  factorizations %d\n' %
                    (results[result+1], iter, gnorm, value, nfactor));
    return x, result, Hfree, free, trace    