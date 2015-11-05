from numpy import *

from boxQP import boxQP


def app_tile(A, reps):
    A_ = A[:]
    if A.ndim < len(reps):
        while len(A_.shape) < len(reps):
            A_ = expand_dims(A_, axis=len(A_.shape))
    return tile(A_, reps)

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
        'lims':           None,  # control limits
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
        'cost':           None,  # initial cost for pre-rolled trajectory
    }

    # --- initial sizes and controls
    n = x0.shape[-1]          # dimension of state vector
    m = u0.shape[1]          # dimension of control vector
    N = u0.shape[0]         # number of state transitions
    u = u0[:]

    # -- process options
    options.update(options_in)

    verbosity = options["print"]

    if options["lims"] is not None:
        len_lims = options["lims"].size
        if len_lims == 0:
            pass
        elif len_lims == 2*m:
            options["lims"] = sort(options["lims"], 1)
        elif len_lims == 2:
            options["lims"] = dot(ones((m,1)), sort(options["lims"].flatten(1)))
        elif len_lims == m:
            options["lims"] = dot(options["lims"].flatten(1), [-1, 1])
        else:
            raise ValueError("limits are of the wrong size")

    lamb = options["lambdaInit"]
    dlamb = options["dlambdaInit"]

    # Initial trajectory
    if x0.shape[0] == n:
        diverge = True
        for alpha in options["Alpha"]:
            xn, un, costn = forward_pass(x0, alpha*u, None, None, None, array([1]), dyncst, options["lims"])
            # simplistic divergence test
            if all(abs(xn) < 1e8):
                u = un[:, 0]
                x = xn[:, 0]
                cost = costn[:, 0]
                diverge = False
                break
    elif x0.shape[0] == N+1: # already did initial fpass
        x = x0.copy()
        diverge = False
        if options["cost"] is None:
            raise ValueError("pre-rolled initial trajectory requires cost")
        else:
            cost = options["cost"]

    if diverge:
        if verbosity > 0:
            print("\nEXIT: Initial control sequence caused divergence\n")
        return x, u, None, None, None, None

    flgChange = 1
    dcost = 0
    z = 0
    expected = 0
    L = zeros((N, n, m))
    if verbosity > 0:
        print("\n============== begin iLQG ===============\n")

    for alg_iter in range(options["maxIter"]):

        # ==== STEP 1: differentiate dynamics along new trajectory
        if flgChange:
            fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu = dyncst(x, vstack((u, full([1, m], nan))), arange(N), True)
            flgChange = 0

        # ==== STEP 2: backward pass, compute optimal control law and cost-to-go
        backPassDone = 0
        while not backPassDone:
            diverge, Vx, Vxx, l, L, dV = back_pass(cx, cu, cxx, cxu, cuu, fx, fu, fxx, fxu, fuu, lamb, options["regType"], options["lims"], u)

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
        g_norm = mean((abs(l) / (abs(u[0])+1)).max(0))
        if g_norm < options["tolGrad"] and lamb < 1e-5:
            dlamb = min(dlamb / options["lambdaFactor"], 1/options["lambdaFactor"])
            lamb = lamb * dlamb * (lamb > options["lambdaMin"])
            if verbosity > 0:
                print("\nSUCCESS: gradient norm < tolGrad\n")
            break

        # ==== STEP 3: line-search to find new control sequence, trajectory, cost
        fwdPassDone = 0
        if backPassDone:
            if options["parallel"]: # parallel line-search
                xnew, unew, costnew = forward_pass(x0, u, L, x[:N], l, options["Alpha"], dyncst, options["lims"])
                dcost = cost.flatten(1).sum(axis=0) - costnew.sum(axis=1)
                w = argmax(dcost)
                dcost = dcost[0, w]
                alpha = options["Alpha"][w]
                expected = -alpha*(dV[0] + alpha*dV[1])
                if expected > 0:
                    z = dcost/expected
                else:
                    z = sign(dcost)
                    print("WARNING: non-positive expected reduction: should not occur")
                if z > options["zMin"]:
                    fwdPassDone = 1
                    costnew = costnew[:,:,w]
                    xnew = xnew[:,:,w]
                    unew = unew[:,:,w]
            else: # serial backtracking line-search
                for alpha in options["Alpha"]:
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
                print('iter: {} cost: {} reduction: {} gradient: {} log10lam: {}'.format(alg_iter, sum(cost.flatten(1)), dcost, g_norm, nan if lamb == 0 else log10(lamb)))

            # decrease lambda
            dlamb = min(dlamb / options["lambdaFactor"], 1/options["lambdaFactor"])
            lamb = lamb * dlamb * (lamb > options["lambdaMin"])

            # accept changes
            u = unew[:,:,None]
            x = xnew[:,:,None]
            cost = costnew[:,:,None]
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
    else:
        print("\nEXIT: Maximum iterations reached.\n")

    return x, u, L, Vx, Vxx, cost


def forward_pass(x0, u, L, x, du, alpha, dyncst, lims):

    n = x0.shape[0]
    K = alpha.shape[0]
    N = u.shape[0]
    m = u.shape[1]

    xnew = zeros((N+1, K, n))
    xnew[0, :, :] = x0
    unew = zeros((N, K, m))
    cnew = zeros((N+1, K))
    for i in range(N):
        unew[i] = u[i]

        if du is not None:
            unew[i] = unew[i] + dot(du[i], alpha)

        if L is not None:
            dx = xnew[i] - x[i]
            unew[i] = unew[i] + dot(L[i], dx)

        if lims is not None:
            unew[:,:,i] = clip(unew[i], lims[0], lims[1])

        xnew[i+1], cnew[i] = dyncst(xnew[i], unew[i], i*ones(K))

    _, cnew[N] = dyncst(xnew[N], full([K, m], nan), i)

    return xnew, unew, cnew


def back_pass(cx, cu, cxx, cxu, cuu, fx, fu, fxx, fxu, fuu, lamb, regType, lims, u):
    """
    Perform the Ricatti-Mayne backward pass
    """

    # tensor multiplication for DDP terms
    # vectens = @(a,b) permute(sum(bsxfun(@times,a,b),1), [3 2 1]);
    vectens = lambda a, b: transpose(sum(a*b, axis=1), [2, 1, 0])

    N = cx.shape[0]
    n = cx.shape[1]
    m = cu.shape[1]

    k = zeros((N-1, m))
    K = zeros((N-1, m, n))
    Vx = zeros((N, n))
    Vxx = zeros((N, n, n))
    dV = array([0, 0])

    Vx[N-1] = cx[N-1]
    Vxx[N-1]  = cxx[N-1]

    diverge = 0

    for i in reversed(range(N-1)):
        Qu = cu[i] + dot(fu[i].conj().T, Vx[i+1, :, None])
        Qx = cx[i] + dot(fx[i].conj().T, Vx[i+1, :, None])

        Qux = cxu[i].conj().T + dot(dot(fu[i].conj().T, Vxx[i+1]), fx[i])
        if fxu is not None:
            fxuVx = vectens(Vx[i+1], fxu[i])
            Qux = Qux + fxuVx

        Quu = cuu[:,:,i] + dot(dot(fu[:,:,i].conj().transpose(), Vxx[:,:,i+1]), fu[:,:,i])
        if fuu is not None:
            fuuVx = vectens(Vx[:,i+1], fuu[:,:,:,i])
            Quu = Quu + fuuVx

        Qxx = cxx[:,:,i] + dot(dot(fx[:,:,i].conj().transpose(), Vxx[:,:,i+1]), fx[:,:,i])
        if fxx is not None:
            fxxVx = vectens(Vx[:,i+1], fxx[:,:,:,i])
            Qxx = Qxx + fxxVx

        Vxx_reg = (Vxx[:,:,i+1] + lamb*eye(n)*(regType==2))

        Qux_reg = cxu[:,:,i].conj().transpose() + dot(dot(fu[:,:,i].conj().transpose(), Vxx_reg), fx[:,:,i])
        if fxu is not None:
            Qux_reg = Qux_reg + fxuVx

        QuuF = cuu[:,:,i] + dot(dot(fu[:,:,i].conj().T, Vxx_reg), fu[:,:,i]) + lamb*eye(m)*(regType == 1)

        if fuu is not None:
            QuuF = QuuF + fuuVx

        if lims is None or lims[0,0] > lims[0, 1]:
            # no control limits: Cholesky decomposition, check for non-PD
            try:
                R = linalg.cholesky(QuuF).T
            except linalg.LinAlgError as e:
                print(e)
                diverge = i
                return diverge, Vx, Vxx, k, K, dV

            # find control law
            val1 = empty([Qu.shape[0], 1])
            val1[:, 0] = Qu
            kK = linalg.solve(-R, linalg.solve(R.conj().transpose(), concatenate((val1, Qux_reg), 1)))
            k_i = kK[:,0]
            K_i = kK[:,1:n+1]

        else:   # Solve Quadratic Program
            lower = lims[:,0]-u[:,i,0]
            upper = lims[:,1]-u[:,i,0]

            k_i, result, R, free = boxQP(QuuF, Qu, lower, upper, k[:,min((i+1, N-2))])
            if result < 1:
                diverge = i
                return diverge, Vx, Vxx, k, K, dV

            K_i = zeros((m, n))
            if any(free):
                Lfree = linalg.solve(-R, linalg.solve(R.conj().T, Qux_reg[free,:]))
                K_i[free,:] = Lfree

        # update cost-to-go approximation
        v1 = dot(k_i.conj().T, Qu)
        v2 = dot(dot(.5*k_i.conj().T, Quu), k_i)
        val = [v1, v2]
        dV = dV + val
        Vx[:,i] = Qx + dot(dot(K_i.conj().T, Quu), k_i) + dot(K_i.conj().T, Qu) + dot(Qux.conj().T, k_i)
        Vxx[:,:,i]  = Qxx + dot(dot(K_i.conj().T, Quu), K_i) + dot(K_i.conj().T, Qux) + dot(Qux.conj().T, K_i)
        Vxx[:,:,i]  = .5*(Vxx[:,:,i] + Vxx[:,:,i].conj().T)

        # save controls/gains
        k[:,i] = k_i
        K[:,:,i] = K_i

    return diverge, Vx, Vxx, k, K, dV