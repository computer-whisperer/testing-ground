from numpy import *
import ilqg
"""
A demo of iLQG/DDP with a control-limited LTI system.
"""

print('A demonstration of the iLQG/DDP algorithm\n'
      'with a random control-limited time-invariant linear system.\n'
      'for details see\nTassa, Mansard & Todorov, ICRA 2014\n'
      '\"Control-Limited Differential Dynamic Programming\"\n')

# make stable linear dynamics
h = .01  # time step
n = 10  # state dimension
m = 2  # control dimension
A = random.randn(n, n)
A = A-A.conj().T  # skew-symmetric = pure imaginary eigenvalues
A = exp(h, A)  # discrete time
B = h*random.randn(n, m)

# quadratic costs
Q = h*eye(n)
R = .1*h*eye(m)

# control limits
# Op.lims = ones(m,1)*[-1 1]*.6;

# optimization problem
dyncst = lambda x, u, i, want_all=False: lin_dyn_cst(x, u, A, B, Q, R, want_all)
T = 1000  # horizon
x0 = random.randn(n, 1)  # initial state
u0 = .1*random.randn(m, T)  # initial controls

# run the optimization
ilqg.ilqg(dyncst, x0, u0, {})

def lin_dyn_cst(x, u, A, B, Q, R, want_all=False):
    # For a PD quadratic u-cost
    # no cost (nans) is equivalent to u=0
    u[isnan(u)] = 0

    if not want_all:
        f = dot(A, x) + dot(B, u)
        c = 0.5*sum(x*(dot(Q,x)),1) + 0.5*sum(u * dot(R, u), 1)
        return f, c
    else:
        N = x.shape(2)
        fx = tile(A, [1, 1, N])
        fu = tile(B, [1, 1, N])
        cx = dot(Q, x)
        cu = dot(R, u)
        cxx = tile(Q, [1, 1, N])
        cxu = tile(zeros(B.shape), [1, 1, N])
        cuu = tile(R, [1, 1, N])
        return None, None, fx, fu, None, None, None, cx, cu, cxx, cxu, cuu

