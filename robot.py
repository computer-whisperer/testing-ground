import wpilib
from pyfrc.physics import drivetrains
from numpy import *
import time


import ilqg


def dynamics_func(x, u, dt=.1):

    # === states and controls:
    # x = [x y r x' y' r']' = [x y r]
    # u = [r l]'     = [right_wheel_out left_wheel_out

    # controls
    l_out = u[0]  # w = right wheel out
    r_out = u[1]  # a = left wheel out

    y_spd, rot_spd = drivetrains.two_motor_drivetrain(l_out, r_out)

    theta = x[2] + dt*rot_spd
    world_vel = array([sin(theta), cos(theta)]) * y_spd

    pos = concatenate((x[:2] + world_vel*dt, theta[None]))

    return pos


def cost_func(x, u):
    # cost function for car-parking problem
    # sum of 3 terms:
    # lu: quadratic cost on controls
    # lf: final cost on distance from target parking configuration
    # lx: small running cost on distance from origin to encourage tight turns

    cu = 1e-2*array([1, 1])         # control cost coefficients

    cf = array([ 1,  1,  1])    # final cost coefficients
    pf = array([.01, .01,  1])   # smoothness scales for final cost

    cx = array([.1,   .1,  1])          # running cost coefficients
    px = .1*ones(3)            # smoothness scales for running cost

    if any(isnan(u)):
        u[:] = 0
        lf = dot(cf, sabs(x[:3], pf).T)
    else:
        lf = 0

    # control cost
    lu = dot(u*u, cu)

    # running cost
    lx = dot(sabs(x[:3], px), cx)

    # total const
    c = lu + lx + lf
    return c


def sabs(x, p):
    # smooth absolute-value function (a.k.a pseudo-Huber)
    return sqrt(x*x + p*p) - p


class IlqgRobot(wpilib.IterativeRobot):

    def robotInit(self):
        # optimization problem
        T = 100              # horizon
        x0 = array([10,  -10,  0])   # initial state
        u0 = .1*random.randn(T, 2)  # initial controls
        #u0 = zeros((T, 2))  # initial controls
        options = {}

        # run the optimization
        options["lims"] = array([[-1, 1],
                                 [-1, 1]])
        start_time = time.time()
        self.x, self.u, L, Vx, Vxx, cost = ilqg.ilqg(lambda x, u: dynamics_func(x, u), cost_func, x0, u0, options)
        self.i = 0
        print(self.x[-1])
        print("ilqg took {} seconds".format(time.time() - start_time))

        self.drive = wpilib.RobotDrive(0, 1)
        self.joystick = wpilib.Joystick(0)

    def autonomousInit(self):
        self.autostart = time.time()

    def autonomousPeriodic(self):
        time_elapsed = time.time() - self.autostart
        if time_elapsed < self.u.shape[0]*.1:
            self.drive.tankDrive(self.u[time_elapsed//.1, 0], -self.u[time_elapsed//.1, 1])
        else:
            self.drive.tankDrive(0, 0)

    def teleopPeriodic(self):
        self.drive.arcadeDrive(self.joystick)


if __name__ == "__main__":
    wpilib.run(IlqgRobot)
