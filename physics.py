from pyfrc.physics import core, drivetrains
from robot import dynamics_func
from numpy import *

class PhysicsEngine(core.PhysicsEngine):

    def __init__(self, physics_controller):
        self.physics_controller = physics_controller

    def update_sim(self, hal_data, now, tm_diff):
        l_motor = hal_data["pwm"][0]["value"]
        r_motor = hal_data["pwm"][1]["value"]

        x = array(self.physics_controller.get_position())
        u = array((l_motor, r_motor))

        xnew = dynamics_func(x, u, tm_diff)
        delta_x = xnew - x
        self.physics_controller._move(delta_x[0], delta_x[1], delta_x[2])
