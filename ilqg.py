from numpy import *
import timeit

def ilqg(dyncst, x0, u0, options_in={}):
    """
    PORTED FROM MATLAB CODE
    http://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization


    iLQG - solve the deterministic finite-horizon optimal control problem.
               minimize sum_i CST(x(:,i),u(:,i)) + CST(x(:,end))
               u
           s.t.  x(:,i+1) = DYN(x(:,i),u(:,i))

    Inputs
    ======
    DYNCST - A combined dynamics and cost function. It is called in
    three different formats.
     1) step:
      [xnew,c] = DYNCST(x,u,i) is called during the forward pass.
      Here the state x and control u are vectors: size(x)==[n 1],
      size(u)==[m 1]. The cost c and time index i are scalars.
      If Op.parallel==true (the default) then DYNCST(x,u,i) is be
      assumed to accept vectorized inputs: size(x,2)==size(u,2)==K

     2) final:
      [~,cnew] = DYNCST(x,nan) is called at the end the forward pass to compute
      the final cost. The nans indicate that no controls are applied.

     3) derivatives:
      [~,~,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = DYNCST(x,u,I) computes the
      derivatives along a trajectory. In this case size(x)==[n N+1] where N
      is the trajectory length. size(u)==[m N+1] with NaNs in the last column
      to indicate final-cost. The time indexes are I=(1:N).
      Dimensions match the variable names e.g. size(fxu)==[n n m N+1]
      note that the last temporal element N+1 is ignored for all tensors
      except cx and cxx, the final-cost derivatives.

    x0 - The initial state from which to solve the control problem.
      Should be a column vector. If a pre-rolled trajectory is available
      then size(x0)==[n N+1] can be provided and Op.cost set accordingly.
    u0 - The initial control sequence. A matrix of size(u0)==[m N]
      where m is the dimension of the control and N is the number of state
      transitions.
    Op - optional parameters, see below

    Outputs
    =======
    x - the optimal state trajectory found by the algorithm.
        size(x)==[n N+1]
    u - the optimal open-loop control sequence.
        size(u)==[m N]
    L - the optimal closed loop control gains. These gains multiply the
        deviation of a simulated trajectory from the nominal trajectory x.
        size(L)==[m n N]
    Vx - the gradient of the cost-to-go. size(Vx)==[n N+1]
    Vxx - the Hessian of the cost-to-go. size(Vxx)==[n n N+1]
    cost - the costs along the trajectory. size(cost)==[1 N+1]
           the cost-to-go is V = fliplr(cumsum(fliplr(cost)))
    lambda - the final value of the regularization parameter
    trace - a trace of various convergence-related values. One row for each
            iteration, the columns of trace are
            [iter lambda alpha g_norm dcost z sum(cost) dlambda]
            see below foe details.
    """

    # user-adjustable parameters
    options = {
        'lims':           [],  # control limits
        'parallel':       True,  # use parallel line-search?
        'Alpha':          10**linspace(0, -3, 8),  # backtracking coefficients
        'tolFun':         1e-7,  # reduction exit criterion
        'tolGrad':        1e-5,  # gradient exit criterion
        'maxIter':        500,  # maximum iterations
        'lambdaInit':     1,  # initial value for lambda
        'dlambdaInit':    1,  # initial value for dlambda
        'lambdaFactor':   1.6,  # lambda scaling factor
        'lambdaMax':      1e10,  # lambda maximum value
        'lambdaMin':      1e-6,  # below this value lambda = 0
        'regType':        1,  # regularization type 1: q_uu+lambda*eye(); 2: V_xx+lambda*eye()
        'zMin':           0,  # minimal accepted reduction ratio
        'plot':           1,  # 0: no;  k>0: every k iters; k<0: every k iters, with derivs window
        'print':          2,  # 0: no;  1: final; 2: iter; 3: iter, detailed
        'cost':           [],  # initial cost for pre-rolled trajectory
    }

    # --- initial sizes and controls
    n = x0.shape[0]          # dimension of state vector
    m = u0.shape[0]          # dimension of control vector
    N = u0.shape[1]         # number of state transitions
    u = u0[:]

    # -- process options
    options.update(options_in)

    verbosity = options["print"]

    len_lims = len(options["lims"])
    if len_lims == 0:
        pass
    elif len_lims == 2*m:
        options["lims"] = sort(options["lims"], 2)
    elif len_lims == 2:
        options["lims"] = dot(ones((m,1)), sort(options["lims"].flatten(1)))
    elif len_lims == m:
        options["lims"] = dot(options["lims"].flatten(1), [-1, 1])
    else:
        raise ValueError("limits are of the wrong size")

    lamb = options["lambdaInit"]
    dlamb = options["dlambdaInit"]

    # Initial trajectory
    if x0.shape[1] == 1:
        diverge = True
        for alpha in options["Alpha"]:
            x, un, cost = forward_pass(x0[:, 0], alpha*u, array([]), array([]), array([]), 1, dyncst, options["lims"])
            # simplistic divergence test
            if all(abs(x[:]) < 1e8):
                u = un
                diverge = False
                break
    elif x0.shape[1] == N+1: # already did initial fpass
        x = x0.copy()
        diverge = False
        if empty(options["cost"]):
            raise ValueError("pre-rolled initial trajectory requires cost")
        else:
            cost = options["cost"]

    if diverge:
        Vx, Vxx, stop = nan
        L = zeros((m, n, N))
        cost = array([])
        timing = array([0, 0, 0, 0])
        trace = array([1, lamb, nan, nan, nan, sum(cost.flatten(1)), dlamb])
        if verbosity > 0:
            print("\nEXIT: Initial control sequence caused divergence\n")
        return x, u, L, Vx, Vxx, cost, trace, stop, timing

    flgChange = 1
    #t_total = tic
    #diff_t = 0
    #back_t = 0
    #fwd_t = 0
    stop = 0
    dcost = 0
    z = 0
    expected = 0
    trace = zeros((min(options["maxIter"], 1e6), 8))
    trace[0,:] = [1, lamb, nan, nan, nan, sum(cost.flatten(1)), dlamb]
    L = zeros((m, n, N))
    if verbosity > 0:
        print("\n============== begin iLQG ===============\n")

    for alg_iter in range(options["maxIter"]):
        if stop:
            break

        # ==== STEP 1: differentiate dynamics along new trajectory
        if flgChange:
            #t_diff = tic
            _, _, fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu = dyncst(x, array([u, array([m, 1]).fill(nan)]), arange(1., N+2), True)
            #diff_t = diff_t + toc(t_diff)
            flgChange = 0

        # ==== STEP 2: backward pass, compute optimal control law and cost-to-go
        backPassDone = 0
        while not backPassDone:
            # t_back = tic
            diverge, Vx, Vxx, l, L, dV = back_pass(cx, cu, cxx, cxu, cuu, fx, fu, fxx, fxu, fuu, lamb, options["regType"], options["lims"], u)
            # back_t = back_t + toc(t_back

            if diverge:
                if verbosity > 2:
                    print("Cholesky failed at timestep {}.".format(diverge))
                dlamb = max(dlamb * options["lambdaFactor"], options["lambdaFactor"])
                lamb = max(lamb * dlamb, options["lambdaMin"])
                if lamb > options["lambdaMax"]:
                    break
                continue
            backPassDone = 1

            #Check for termination due to small gradient
            g_norm = mean(max(abs(l) / (abs(u)+1),[],1))
            trace[alg_iter][0] = alg_iter
            trace[alg_iter][4] = g_norm
            trace[alg_iter][7] = nan
            if g_norm < options["tolGrad"] and lamb < 1e-5:
                dlamb = min(dlamb / options["lambdaFactor"], 1/options["lambdaFactor"])
                lamb = lamb * dlamb * (lamb > options["lambdaMin"])
                trace[alg_iter][2] = lamb
                trace[alg_iter][8] = dlamb
                if verbosity > 0:
                    print("\nSUCCESS: gradient norm < tolGrad\n")
                break

        # ==== STEP 3: line-search to find new control sequence, trajectory, cost
        fwdPassDone = 0
        if backPassDone:
            #t_fwd = tic
            if options["parallel"]: # parallel line-search
                xnew, unew, costnew = forward_pass(x0, u, L, x[:,0:N], l, options["alpha"], dyncst, options["lims"])
                dcost = sum(cost.flatten(1)) - sum(costnew, 2)
                dcost, w = max(dcost)
                alpha = options["alpha"](w)
                expected = -alpha*(dV(1) + alpha*dV(2))
                if expected > 0:
                    z = dcost/expected
                else:
                    z = sign(dcost)
                    print("WARNING: non-positive expected reduction: should not occur")
                if z > options["zMin"]:
                    fwdPassDone = 1
                    costnew = costnew[:,:,w-1]
                    xnew = xnew[:,:,w-1]
                    unew = unew[:,:,w-1]
            else: # serial backtracking line-search
                for alpha in options["alpha"]:
                    xnew, unew, costnew = forward_pass(x0, u+l*alpha, L, x[:,0:N], [], 1, dyncst, options["lims"])
                    dcost = sum(cost.flatten(1)) - sum(costnew.flatten(1))
                    expected = -alpha*(dV(1) + alpha*dV(2))
                if expected > 0:
                    z = dcost/expected
                else:
                    z = sign(dcost)
                    print("WARNING: non-positive expected reduction: should not occur")
                if z > options["zMin"]:
                    fwdPassDone = 1
                    break
            # fwd_t = fwd_t + toc(t_fwd)

        # ==== STEP 4: accept (or not)
        if fwdPassDone:

            # print status
            if verbosity > 1:
                print('iter: {} cost: {} reduction: {} gradient: {} log10lam: {}'.format(alg_iter, lamb, g_norm, log10(dlamb)))

            # decrease lambda
            dlamb = min(dlamb / options["lambdaFactor"], 1/options["lambdaFactor"])
            lamb = lamb * dlamb * (lamb > options["lambdaFactor"])

            # accept changes
            u = unew
            x = xnew
            cost = costnew
            flgChange = 1

            # update trace
            trace[alg_iter,:] = [alg_iter, lamb, alpha, g_norm, dcost, z, sum(cost.flatten(1)), dlamb]

            # terminate ?
            if dcost < options["tolFun"]:
                if verbosity > 0:
                    print("\nSUCCESS: cost change < tolFun")
                break

        else: # No cost improvement

            # increase lambda
            dlamb = max(dlamb * options["lambdaFactor"], options["lambdaFactor"])
            lamb = max(lamb * dlamb, options["lambdaMin"])

            # print status
            if verbosity > 1:
                print('iter: {} REJECTED expected: {} actual: {} log10lam: {}'.format(alg_iter, expected, dcost, log10(dlamb)))

            # update trace
            trace[alg_iter,:] = [alg_iter, lamb, alpha, g_norm, dcost, z, sum(cost.flatten(1)), dlamb]

            # terminate ?
            if lamb > options["lambdaMax"]:
                if verbosity > 0:
                    print("\nEXIT: lambda > lambdaMax")
                break
        break
    else:
        print("\nEXIT: Maximum iterations reached.\n")

    return x, u, L, Vx, Vxx, cost, trace


def forward_pass(x0, u, L, x, du, alpha, dyncst, lims):
    """
    parallel forward-pass (rollout)
    internally time is on the 3rd dimension
    to facilitate vectorized dynamics calls
    """

    n = x0.shape[0]
    s = shape(alpha)
    if len(s) == 0:
        K = 1
    else:
        K = max(s)
    K1 = ones((1, K)) # useful for expansion
    m = u.shape[0]
    N = u.shape[1]

    xnew = zeros((n, K, N))
    xnew[:,:,0] = x0
    unew = zeros((m, K, N))
    cnew = zeros((1, K, N+1))
    for i in range(N):
        unew[:,:,i] = u[:,i*K1]

        if du is not None:
            unew[:,:,i] = unew[:,:,i] + du[:,i]*alpha

        if L is not None:
            dx = xnew[:,:,i] - x[:,i*K1]
            unew[:,:,i] = unew[:,:,i] + L[:,:,i]*dx

        if lims is not None:
            unew[:,:,i] = min(lims[:,2*K1], max(lims[:,1*K1], unew[:,:,i]))

        xnew[:,:,i+1], cnew[:,:,i] = dyncst(xnew[:,:,i], unew[:,:,i], i*K1)
    _, cnew[:,:,i] = dyncst(xnew[:,:,N+1], array([m, K, 1]).fill(nan))

    # put the time dimension in the columns
    xnew = xnew.transpose[1, 3, 2]
    unew = unew.transpose[1, 3, 2]
    cnew = cnew.transpose[1, 3, 2]

    return xnew, unew, cnew

def back_pass(cx, cu, cxx, cxu, cuu, fx, fu, fxx, fxu, fuu, lamb, regType, lims, u):
    """
    Perform the Ricatti-Mayne backward pass
    """

    # tensor multiplication for DDP terms
    vectens = lambda a, b: transpose(sum(a*b, 1), [3, 2, 1])

    N = cx.shape[1]
    n = len(cx)/N
    m = len(cu)/N

    cx    = reshape(cx,  [n, N])
    cu    = reshape(cu,  [m, N])
    cxx   = reshape(cxx, [n, n, N])
    cxu   = reshape(cxu, [n, m, N])
    cuu   = reshape(cuu, [m, m, N])

    k     = zeros((m,N-1))
    K     = zeros((m,n,N-1))
    Vx    = zeros((n,N))
    Vxx   = zeros((n,n,N))
    dV    = [0, 0]

    Vx[:,N]     = cx[:,N]
    Vxx[:,:,N]  = cxx[:,:,N]

    diverge = 0

    for i in reversed(range(N)):

        Qu = cu[:,i] + fu[:,:,i].conj().transpose()*Vx[:,i+1]
        Qx = cx[:,i] + fx[:,:,i].conj().transpose()*Vx[:,i+1]

        Qux = cxu[:,:,i] + fu[:,:,i].conj().transpose()*Vxx[:,:,i+1]*fx[:,:,i]
        if fxu is not None:
            fxuVx = vectens(Vx[:,i+1], fxu[:,:,:,i])
            Qux = Qux + fxuVx

        Quu = cuu[:,:,i] + fu[:,:,i].conj().transpose()*Vxx[:,:,i+1]*fu[:,:,i]
        if fuu is not None:
            fuuVx = vectens(Vx[:,i+1], fuu[:,:,:,i])
            Quu = Quu + fuuVx

        Qxx = cxx[:,:,i] + fx[:,:,i].conj().transpose()*Vxx[:,:,i+1]*fx[:,:,i]
        if fxx is not None:
            fxxVx = vectens(Vx[:,i+1], fxx[:,:,:,i])
            Qxx = Qxx + fxxVx

        Vxx_reg = (Vxx[:,:,i+1] + lamb*eye(n)*(regType==2))

        Qux_reg = cxu[:,:,i].conj().transpose() + fu[:,:,i].conj().transpose()*Vxx_reg*fx[:,:,i]
        if fxu is not None:
            Qux_reg = Qux_reg + fxuVx

        QuuF = cuu[:,:,i] + fu[:,:,i].conj().T*Vxx_reg*fu[:,:,i] + lamb*eye(m)*(regType == 1)

        if fuu is not None:
            QuuF = QuuF + fuuVx

        if lims is not None or lims[0,0] > lims[0, 1]:
            # no control limits: Cholesky decomposition, check for non-PD
            try:
                R = linalg.cholesky(QuuF).T
            except linalg.LinAlgError as e:
                print(e)
                diverge = i
                return diverge, Vx, Vxx, k, K, dV

            # find control law
            kK = linalg.solve(-R, linalg.solve(R.conj().transpose(), [Qu, Qux_reg]))
            k_i = kK[:,1]
            K_i = kK[:,2:n+1]

        else:   # Solve Quadratic Program
            lower = lims[:,1]-u[:,i]
            upper = lims[:,2]-u[:,i]

            k_i, result, R, free = boxQP(Quuf, Qu, lower, upper, k[:,min((i+1, N-1))])
            if result < 1:
                diverge = i
                return diverge, Vx, Vxx, k, K, dV

            K_i = zeros((m, n))
            if any(free):
                Lfree = linalg.solve(-R, linalg.solve(R.conj().T, Qux_reg[free,:]))
                K_i[free,:] = Lfree

        # update cost-to-go approximation
        dV = dV + [k_i.conj().T*Qu, .5*k_i.conj().T*Quu*k_i]
        Vx[:,i] = Qx + K_i.conj().T*Quu*k_i + K_i.conj().T*Qu + Qux.conj().T*k_i
        Vxx[:,:,i]  = Qxx + K_i.conj().T*Quu*K_i + K_i.conj().T*Qux + Qux.conj().T*K_i
        Vxx[:,:,i]  = .5*(Vxx[:,:,i] + Vxx[:,:,i].conj().T)

        # save controls/gains
        k[:,i] = k_i
        K[:,:,i] = K_i

        return diverge, Vx, Vxx, k, K, dV