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

def boxQP(H, g, lower, upper, x0=None, options_in=None):
    
    n        = H.shape[0]
    clamped  = full((n, 1), False)
    free     = full((n, 1), True)
    oldvalue = 0
    result   = 0
    gnorm    = 0
    nfactor  = 0
    trace    = []
    Hfree    = zeros(n)

    # initial state
    if x0 is not None and x0.size == n:
        x = clip(x0, lower, upper)
    else:
        LU = [lower, upper]
        LU[not isfinite(LU)] = nan
        x = nanmean(LU,2)
    x[isinf(x)] = 0
    
    # options
    options = {
        "maxIter": 100,     # maximum number of iterations
        "minGrad": 1e-8,     # minimum norm of non-fixed gradient
        "minRelImprove": 1e-8,     # minimum relative improvement
        "stepDec": 0.6,     # factor for decreasing stepsize
        "minStep": 1e-22,     # minimal stepsize for linesearch
        "Armijo": 0.1, 	# Armijo parameter (fraction of linear improvement required)
        "verbose":  0, # verbosity
    }
    if options_in is not None:
        options.update(options_in)
    
    # initial objective value
    value    = dot(x.conj().T, g) + 0.5*dot(dot(x.conj().T, H),x)
    
    if options["verbose"] > 0:
        print('==========\nStarting box-QP, dimension #-3d, initial value: #-12.3f\n',n, value)
    
    # main loop
    for iter in range(options["maxIter"]):
        
        if result !=0:
            break
        
        # check relative improvement
        if iter>0 and all((oldvalue - value)) < all(options["minRelImprove"]*abs(oldvalue)):
            result = 4
            break
        oldvalue = value
        
        # get gradient
        grad     = g + dot(H, x)
        
        # find clamped dimensions
        old_clamped                     = clamped
        clamped                         = full((n,1), False, dtype=bool)
        clamped[((x == lower)&(grad>0))]  = True
        clamped[((x == upper)&(grad<0))]  = True
        free                            = logical_not(clamped)
        
        # check for all clamped
        
        if all(clamped):
            result = 6
            break
        
        # factorize if clamped has changed
        if iter == 1:
            factorize    = True
        else:
            factorize    = any(old_clamped != clamped)
        
        if factorize:
            try:
                val1 = H[free[:, 0],:][:,free[:, 0]]
                Hfree = linalg.cholesky(val1).T
            except linalg.LinAlgError as e:
                print(e)
                result = -1
                break
            nfactor += 1
        
        # check gradient norm
        gnorm  = linalg.norm(grad[free.flatten(1)])
        if gnorm < options["minGrad"]:
            result = 5
            break
        
        # get search direction
        grad_clamped   = g  + dot(H, (x*clamped))
        search         = zeros((n,1))
        search[free]   = linalg.solve(-Hfree, linalg.solve(Hfree.conj().T, grad_clamped[free])) - x[free]
        
        # check for descent direction
        sdotg          = sum(search*grad)
        if sdotg >= 0: # (should not happen)
            break
        
        # armijo linesearch
        step  = 1
        nstep = 0
        xc    = clip(x+step*search, lower, upper)
        vc    = xc.conj().T*g + 0.5*xc.conj().T*H*xc
        while any((vc - oldvalue)/(step*sdotg)) < options["Armijo"]:
            step  = step*options["stepDec"]
            nstep = nstep+1
            xc    = clip(x+step*search, lower, upper)
            vc    = xc.conj().T*g + 0.5*xc.conj().T*H*xc
            if step<options["minStep"]:
                result = 2
                break
        
        if options["verbose"] > 1:
            print('iter #-3d  value # -9.5g |g| #-9.3g  reduction #-9.3g  linesearch #g^#-2d  n_clamped #d\n', 
                iter, vc, gnorm, oldvalue-vc, options["stepDec"], nstep, sum(clamped))
        
        # accept candidate
        x     = xc
        value = vc
    
    if iter >= options["maxIter"]:
        result = 1
    
    results = { 'Hessian is not positive definite',          # result = -1
                'No descent direction found',                # result = 0    SHOULD NOT OCCUR
                'Maximum main iterations exceeded',          # result = 1
                'Maximum line-search iterations exceeded',   # result = 2
                'No bounds, returning Newton point',         # result = 3
                'Improvement smaller than tolerance',        # result = 4
                'Gradient norm smaller than tolerance',     # result = 5
                'All dimensions are clamped'}                  # result = 6
    
    if options["verbose"] > 0:
        print('RESULT: {}\niterations {}  gradient {} final value {}  factorizations {}\n'.format(
            result, iter, gnorm, value, nfactor))

    return x, result, Hfree, free

def demoQP():
    n 		= 500
    g 		= random.randn(n,1)
    H 		= random.randn(n,n)
    H 		= dot(H,H.conj().T)
    lower 	= -ones((n,1))
    upper 	=  ones((n,1))
    out = boxQP(H, g, lower, upper, random.randn(n,1))
    print(out)

#demoQP()