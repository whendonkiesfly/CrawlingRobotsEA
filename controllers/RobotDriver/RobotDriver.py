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



# create the Robot instance.
robot = controller.Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

motors = []
# position_sensors = []##todo: maybe do this.
for i in range(robot.getNumberOfDevices()):
    device = robot.getDeviceByIndex(i)
    if isinstance(device, controller.Motor):
        motors.append(device)

def get_motor_range(m):
    r = (m.getMinPosition(), m.getMaxPosition())
    if r[0] == 0 and r[1] == 0:
        r = (-2*math.pi, 2*math.pi)###############################TODO: WHY ARE WE GETTING HERE FOR SALAMANDER LEGS? INFINITE ROTATION?
    adjusted_range = (scale(0.001, (0,1), r), scale(0.999, (0,1), r))
    return adjusted_range

def set_joints(motors, joint_positions):##########################################################################TODO: THIS NOW IS A FORCE CONTROLLER. CHANGE NAME AND STUFF. NEED TO MAKE THE NN ONLY RETURN BETWEEN -1 AND 1. USE THE COMMENTED OUT ACTIVATION FUNCTION.
    assert len(motors) == len(joint_positions), f"Invalid joint count: Expected {len(motors)} but got {len(joint_positions)}."

    positions = [scale(pos, (0, 1), get_motor_range(motor)) for pos, motor in zip(joint_positions, motors)]
    for i in range(len(motors)):
        motors[i].setPosition(positions[i])

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

    joint_inputs = [scale(motor.getTargetPosition(), get_motor_range(motor), (0, 1)) for motor in motors]


    inputs = [timer_input] + joint_inputs
    set_joints(motors, net.process_network(inputs))

# Enter here exit cleanup code.
