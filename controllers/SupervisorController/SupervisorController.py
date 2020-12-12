"""SupervisorController controller."""

import datetime
import json
import os
import random
import sys
import time

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
import controller


###todo: seed? does this affect the other libraries?
random.seed(4)


sys.path.insert(0, os.path.join(os.path.split(os.path.realpath(__file__))[0], ".."))
from ennlib import EASupervisor






ROBOT_CONTROLLER_NAME = "RobotDriver"

# create the Robot instance.
supervisor = Supervisor()

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())




#https://www.cyberbotics.com/doc/reference/supervisor#wb_supervisor_field_import_mf_node

class RobotContainer:
    def __init__(self, robot, start_coord, start_rotation, net_wrapper=None):
        self.robot = robot
        self.start_coord = start_coord
        self.start_rotation = start_rotation
        self.net_wrapper = net_wrapper







def initialize_robots(robot_count, robot_string, robot_offset, z_offset):
    robot_containers = []
    root_children = supervisor.getRoot().getField("children")
    for i in range(robot_count):
        def_name = f"ROBOT_{i}"
        def_str = f"DEF {def_name} {robot_string}"
        root_children.importMFNodeFromString(0, def_str)
        robot = supervisor.getFromDef(def_name)####################################TODO: MAYBE ONLY HAVE ONE ROBOT AND USE THE DEF TO COPY IT AND USE NAME TO ADDRESS IT.
        if robot is None:
            raise RuntimeError("Robot not found after creation. Was there a DEF in the robot string? There shouldn't be one.")

        # robot.getField("controller").setSFString(ROBOT_CONTROLLER_NAME)
        start_coord = [(i-robot_count/2) * robot_offset, 0, z_offset]
        start_rotation = robot.getField("rotation").getSFRotation()
        robot_containers.append(RobotContainer(robot, start_coord, start_rotation))
    return robot_containers

def pause():
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)

def run():
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_RUN)####TODO: MAKE OPTION FOR FAST.


def prepare_cycle(robot_containers, net_wrappers):
    assert len(robot_containers) == len(net_wrappers), f"Number of robots and networks must match. {len(robot_containers)}!={len(net_wrappers)}"
    for i, container in enumerate(robot_containers):
        container.net_wrapper = net_wrappers[i]
        robot = container.robot
        trans_field = robot.getField("translation")
        args_field = robot.getField("controllerArgs")
        rot_field = robot.getField("rotation")
        trans_field.setSFVec3f(container.start_coord)  # Reset start coordinate.
        rot_field.setSFRotation(container.start_rotation)  # Reset rotation.
        nn_string = json.dumps(net_wrappers[i].net.to_dict())

        #Set the new network for the argument.
        #If it doesn't have an arguments array yet, add one. Otherwise just modify it.
        if args_field.getCount() == 0:
            args_field.insertMFString(0, nn_string)
        else:
            args_field.setMFString(0, nn_string)

        robot.restartController()###todo: reset motors?
        robot.resetPhysics()


def update_fitnesses(containers):
    for i, container in enumerate(containers):
        robot = container.robot
        end_coord = robot.getField("translation").getSFVec3f()
        container.net_wrapper.fitness = sum((end_coord[i] - container.start_coord[i])**2 for i in range(3))#########todo: not sure about this.


def count_robot_motors(robot):
    print(type(robot))
    print(dir(robot))
    count = 0
    for i in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(i)
        if isinstance(device, controller.Motor):
            count += 1
    return count



#TODO: FIGURE OUT HOW TO INITIALIZE THE JOINT POSITIONS MAYBE.

def fitness_function_callback(new_population, epoch_time):#############################################!#!#!#!#!!#!#!#!#!#!#!#!#population is nn,fitness pairs!!! need to change everything downstream from here.
    global total_best_fitness
    prepare_cycle(containers, new_population)
    run()

    print("starting fitness")
    start_time = supervisor.getTime()
    while supervisor.getTime() < (start_time + epoch_time):
        if supervisor.step(timestep) == -1:
            sys.exit()############################TODO: CLEANUP OR SOMETHING??? MAYBE RETURN A SPECIAL VALUE FOR CLEANUP.

    pause()

    update_fitnesses(containers)



if __name__ == "__main__":


    weight_min_max = (-2, 2)
    bias_min_max = (-2, 2)
    new_generation_size = 20
    motor_count = 10
    epoch_time = 5
    robot_offset = 3
    layer_count = 3
    z_offset = 0
    max_mutation_count = 20
    robot_type_name = "Salamander"
    output_path = r"c:\temp\blah.txt"




    robot_str = r'{} {{controller "RobotDriver"}}'.format(robot_type_name)





    print("Starting supervisor")
    pause()
    containers = initialize_robots(new_generation_size, robot_str, robot_offset, z_offset)
    nn_output_count = motor_count
    nn_input_count = nn_output_count + 1
    ea = EASupervisor(fitness_function_callback, new_generation_size, nn_input_count, nn_output_count, weight_min_max, bias_min_max, layer_count=layer_count, max_mutation_count=max_mutation_count, callback_args=[epoch_time])
    ea.run()
