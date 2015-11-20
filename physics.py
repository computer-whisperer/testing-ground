from pyfrc.physics import core, drivetrains
from numpy import *

class PhysicsEngine(core.PhysicsEngine):

    def __init__(self, physics_controller):
        self.physics_controller = physics_controller

    def update_sim(self, hal_data, now, tm_diff):
        l_motor = hal_data["pwm"][0]["value"]
        r_motor = hal_data["pwm"][1]["value"]

        y_speed, rot_speed = drivetrains.two_motor_drivetrain(l_motor, r_motor)
        self.physics_controller.drive(y_speed, rot_speed, tm_diff)