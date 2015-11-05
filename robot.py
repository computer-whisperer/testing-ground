import wpilib
from pyfrc.physics import drivetrains
from numpy import *

import ilqg


def finite_difference(fun, x, h=2e-14):
    # simple finite-difference derivatives
    # assumes the function fun() is vectorized

    K, n = x.shape
    H = vstack((zeros(n), h*eye(n)))
    X = x[:, None, :] + H
    X = X.reshape((K*(n+1), n))
    Y = fun(X)
    Y = Y.reshape((K, n+1, -1))
    J = (Y[:, 1:] - Y[:, 0:1]) / h
    return J


def dyn_cst(x, u, want_all=False):
    # combine car dynamics and cost
    # use helper function finite_difference() to compute derivatives

    if not want_all:
        f = dynamics(x, u)
        c = cost(x, u)
        return f, c
    else:

        # dynamics derivatives
        xu_dyn = lambda xu: dynamics(xu[:, 0:3], xu[:, 3:5])
        J = finite_difference(xu_dyn, hstack((x, u)))
        fx = J[:, 0:3]
        fu = J[:, 3:5]

        xu_Jdyn = lambda xu: finite_difference(xu_dyn, xu)
        JJ = finite_difference(xu_Jdyn, hstack((x, u)))
        JJ = JJ.reshape([J.shape[0], J.shape[1], J.shape[1], -1])
        JJ = 0.5*(JJ + JJ.transpose([0, 2, 1, 3]))
        fxx = JJ[:, 0:3, 0:3, :]
        fxu = JJ[:, 0:3, 3:5, :]
        fuu = JJ[:, 3:5, 3:5, :]

        # cost first derivatives
        xu_cost = lambda xu: cost(xu[:, 0:3], xu[:, 3:5])
        J = finite_difference(xu_cost, hstack((x, u)))
        cx = J[:, 0:3, 0]
        cu = J[:, 3:5, 0]

        # cost second derivatives
        xu_Jcst = lambda xu: finite_difference(xu_cost, xu)
        JJ = finite_difference(xu_Jcst, hstack((x, u)))
        cxx = JJ[:, 0:3, 0:3]
        cxu = JJ[:, 0:3, 3:5]
        cuu = JJ[:, 3:5, 3:5]

        return fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu


def dynamics(x, u):

    # === states and controls:
    # x = [x y r]' = [x y r]
    # u = [r l]'     = [right_wheel_out left_wheel_out]

    # constants
    h = 0.03     # h = timestep (seconds)

    # controls
    r_out = u[:, 0]  # w = right wheel out
    l_out = u[:, 1]  # a = left wheel out

    vel, rot = drivetrains.two_motor_drivetrain(l_out, r_out)

    r = x[:, 2] + h*rot  # r = car angle
    z = vstack((cos(r), sin(r))) * h*vel

    dy = vstack([z[0], z[1], rot]).T  # change in state
    y = x + dy  # new state
    return y


def cost(x, u):
    # cost function for car-parking problem
    # sum of 3 terms:
    # lu: quadratic cost on controls
    # lf: final cost on distance from target parking configuration
    # lx: small running cost on distance from origin to encourage tight turns

    final = isnan(u[:, 0])
    u[final, :] = 0

    cu = 1e-2         # control cost coefficients

    cf = array([.1,  .1,   1])  # final cost coefficients
    pf = array([.01, .01, .01]).conj().T  # smoothness scales for final cost

    cx = 1e-3*array([1, 1, 1])  # running cost coefficients
    px = array([.1, .1, .1]).conj().T  # smoothness scales for running cost

    # control cost
    lu = sum(cu*u*u, axis=1)

    # final cost
    if any(final):
        llf = cf * sabs(x[final], pf)
        lf = real(final)
        lf[final] = llf
    else:
        lf = 0

    # running cost
    lx = sum(cx * sabs(x[:, 0:3], px), axis=1)

    # total cost
    c = lu + lx + lf
    return c


def sabs(x,p):
    # smooth absolute-value function (a.k.a pseudo-Huber)
    return sqrt(x**2 + p**2) - p

class IlqgRobot(wpilib.IterativeRobot):

    def robotInit(self):
        # optimization problem
        DYNCST  = lambda x, u, i, want_all=False: dyn_cst(x, u, want_all)
        T       = 50              # horizon
        x0      = array([1, 0, 0])   # initial state
        u0      = .1*random.randn(T, 2)  # initial controls
        options = {}

        # run the optimization
        options["maxIter"] = 5
        x, u = ilqg.ilqg(DYNCST, x0, u0, options)

    def autonomousInit(self):
        pass

    def autonomousPeriodic(self):
        pass


if __name__ == "__main__":
    wpilib.run(IlqgRobot)
