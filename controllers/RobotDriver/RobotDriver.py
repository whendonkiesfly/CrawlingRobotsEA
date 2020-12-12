"""RobotDriver controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

import os
import math
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.split(os.path.realpath(__file__))[0], ".."))
from ennlib import BasicNeuralNet


import controller

def scale(val, in_range, out_range):
    """
    in_range should be of format (in_min, in_max)
    out_range should be of format (out_min, out_max)
    """
    ret = (val-in_range[0])/(in_range[1]-in_range[0])*(out_range[1]-out_range[0])+out_range[0]
    # print(val, ret, in_range, out_range)
    return ret


class MotorInfo:
    POSITION = 1
    VELOCITY = 2

    def __init__(self, motor):
        self.motor = motor
        pos_range = (motor.getMinPosition(), motor.getMaxPosition())
        if pos_range[0] == 0 and pos_range[1] == 0:
            self.mode = self.VELOCITY
            max_v = motor.getMaxVelocity()
            motor.setPosition(float("inf"))
            range = (-max_v, max_v)
        else:
            self.mode = self.POSITION
            range = pos_range
        self.range = (scale(0.001, (0,1), range), scale(0.999, (0,1), range))


# create the Robot instance.
robot = controller.Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

motors = []
# position_sensors = []##todo: maybe do this.
for i in range(robot.getNumberOfDevices()):
    device = robot.getDeviceByIndex(i)
    if isinstance(device, controller.Motor):
        motors.append(MotorInfo(device))


def set_joint_values(motor_infos, joint_values):##########################################################################TODO: THIS NOW IS A FORCE CONTROLLER. CHANGE NAME AND STUFF. NEED TO MAKE THE NN ONLY RETURN BETWEEN -1 AND 1. USE THE COMMENTED OUT ACTIVATION FUNCTION.
    assert len(motor_infos) == len(joint_values), f"Invalid joint count: Expected {len(motor_infos)} but got {len(joint_values)}."

    for motor_info, joint_value in zip(motor_infos, joint_values):
        joint_value = scale(joint_value, (0, 1), motor_info.range)

        if motor_info.mode == MotorInfo.POSITION:
            motor_info.motor.setPosition(joint_value)

        elif motor_info.mode == MotorInfo.VELOCITY:
            motor_info.motor.setVelocity(joint_value)

        else:
            raise Exception("Invalid motor mode:", motor_info.mode)

def get_joint_values(motors):
    get_val = lambda motor_info: motor_info.motor.getTargetPosition() if motor_info.mode is MotorInfo.POSITION else motor_info.motor.getVelocity()
    return [scale(get_val(motor_info), motor_info.range, (0, 1)) for motor_info in motors]


if len(sys.argv) < 2 or sys.argv[1] == "":
    print("No network received in parameter.")
    sys.exit(0)
else:
    net = BasicNeuralNet.from_dict(json.loads(sys.argv[1]))



# Main loop:
MOVE_PERIOD_TIME_SECS = 1
CLOCK_VAL_AMPLITUDE = 1
while robot.step(timestep) != -1:

    cycle_start_time = robot.getTime()
    timer_input = ((cycle_start_time)%MOVE_PERIOD_TIME_SECS) * CLOCK_VAL_AMPLITUDE

    joint_inputs = get_joint_values(motors)

    inputs = [timer_input] + joint_inputs
    set_joint_values(motors, net.process_network(inputs))

# Enter here exit cleanup code.
