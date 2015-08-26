from numpy import *
#function [x,result,Hfree,free,trace] = boxQP(H,g,lower,upper,x0,options)
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

def boxQP(h, g, lower, upper, x0, options):
    
    n        = size(H,1)
    clamped  = false(n,1)
    free     = true(n,1)
    oldvalue = 0
    result   = 0
    gnorm    = 0
    nfactor  = 0
    trace    = []
    Hfree    = zeros(n)
    clamp    = lambda x: max(lower, min(upper, x))
    
    # initial state
    if nargin > 4 and numel(x0)==n:
        x = clamp(x0.flatten(1))
    else:
        LU = [lower, upper]
        LU[not isfinite(LU)] = nan
        x = nanmean(LU,2)
    x[not isfinite(x)] = 0
    
    # options
    if nargin > 5:
        options        = num2cell(options.flatten(1))
        maxIter, minGrad, minRelImprove, stepDec, minStep, Armijo, verbosity = deal(options{:})
    else: # defaults
        maxIter        = 100       # maximum number of iterations
        minGrad        = 1e-8      # minimum norm of non-fixed gradient
        minRelImprove  = 1e-8      # minimum relative improvement
        stepDec        = 0.6       # factor for decreasing stepsize
        minStep        = 1e-22     # minimal stepsize for linesearch
        Armijo         = 0.1   	# Armijo parameter (fraction of linear improvement required)
        verbose          = 0			# verbosity
    
    # initial objective value
    value    = x.conj().T*g + 0.5*x.conj().T*H*x
    
    if verbose > 0:
        print('==========\nStarting box-QP, dimension #-3d, initial value: #-12.3f\n',n, value)
    
    # main loop
    for iter in range(maxIter):
        
        if result !=0:
            break
        
        # check relative improvement
        if iter>0 and (oldvalue - value) < minRelImprove*abs(oldvalue):
            result = 4
            break
        oldvalue = value
        
        # get gradient
        grad     = g + H*x
        
        # find clamped dimensions
        old_clamped                     = clamped
        clamped                         = false(n,1)
        clamped((x == lower)&(grad>0))  = true
        clamped((x == upper)&(grad<0))  = true
        free                            = ~clamped
        
        # check for all clamped
        if all(clamped):
            result = 6
            break
        
        # factorize if clamped has changed
        if iter == 1:
            factorize    = true
        else:
            factorize    = any(old_clamped != clamped)
        
        if factorize:
            [Hfree, indef]  = chol(H(free,free))
            if indef:
                result = -1
                break
            nfactor            = nfactor + 1
        
        # check gradient norm
        gnorm  = norm(grad(free))
        if gnorm < minGrad:
            result = 5
            break
        
        # get search direction
        grad_clamped   = g  + H*(x.*clamped)
        search         = zeros(n,1)
        search(free)   = -Hfree\(Hfree'\grad_clamped(free)) - x(free)
        
        # check for descent direction
        sdotg          = sum(search.*grad)
        if sdotg >= 0: # (should not happen)
            break
        
        # armijo linesearch
        step  = 1
        nstep = 0
        xc    = clamp(x+step*search)
        vc    = xc'*g + 0.5*xc'*H*xc
        while (vc - oldvalue)/(step*sdotg) < Armijo:
            step  = step*stepDec
            nstep = nstep+1
            xc    = clamp(x+step*search)
            vc    = xc'*g + 0.5*xc'*H*xc
            if step<minStep:
                result = 2
                break
        
        if verbose > 1:
            print('iter #-3d  value # -9.5g |g| #-9.3g  reduction #-9.3g  linesearch #g^#-2d  n_clamped #d\n', 
                iter, vc, gnorm, oldvalue-vc, stepDec, nstep, sum(clamped))
        
        if nargout > 4:
            trace(iter).x        = x ##ok<*AGROW>
            trace(iter).xc       = xc
            trace(iter).value    = value
            trace(iter).search   = search
            trace(iter).clamped  = clamped
            trace(iter).nfactor  = nfactor
        
        # accept candidate
        x     = xc
        value = vc
    
    if iter >= maxIter:
        result = 1
    
    results = { 'Hessian is not positive definite',          # result = -1
                'No descent direction found',                # result = 0    SHOULD NOT OCCUR
                'Maximum main iterations exceeded',          # result = 1
                'Maximum line-search iterations exceeded',   # result = 2
                'No bounds, returning Newton point',         # result = 3
                'Improvement smaller than tolerance',        # result = 4
                'Gradient norm smaller than tolerance',     # result = 5
                'All dimensions are clamped'}                  # result = 6
    
    if verbose > 0:
        print('RESULT: #s.\niterations #d  gradient #-12.6 final value #-12.6g  factorizations #d\n',
            results{result+2}, iter, gnorm, value, nfactor)

def demoQP():
    options = [100, 1e-8, 1e-8, 0.6, 1e-22, 0.1, 2] # defaults with detailed printing
    n 		= 500
    g 		= randn(n,1)
    H 		= randn(n,n)
    H 		= H*H.conj().T
    lower 	= -ones(n,1)
    upper 	=  ones(n,1)
    boxQP(H, g, lower, upper, randn(n,1), options)
