"""SupervisorController controller."""

import argparse
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


sys.path.insert(0, os.path.join(os.path.split(os.path.realpath(__file__))[0], ".."))
from ennlib import EASupervisor






ROBOT_CONTROLLER_NAME = "RobotDriver"

# create the Robot instance.
supervisor = Supervisor()

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())




#https://www.cyberbotics.com/doc/reference/supervisor#wb_supervisor_field_import_mf_node

class RobotContainer:
    def __init__(self, robot, start_coord, net_wrapper=None):
        self.robot = robot
        self.start_coord = start_coord
        self.net_wrapper = net_wrapper






###TODO: MAKE SURE ALL ROBOTS ARE THE SAME!
def initialize_robots():
    robot_containers = []
    root_children = supervisor.getRoot().getField("children")
    for i in range(root_children.getCount()):
        node = root_children.getMFNode(i)
        if node.getType() == node.ROBOT and not node.getField("supervisor").getSFBool():
            #We found a robot.
            start_coord = node.getField("translation").getSFVec3f()
            robot_containers.append(RobotContainer(node, start_coord))
    return robot_containers

def pause():
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)

def run():
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_RUN)


def prepare_cycle(robot_containers, net_wrappers):
    assert len(robot_containers) == len(net_wrappers), f"Number of robots and networks must match. {len(robot_containers)}!={len(net_wrappers)}"
    for i, container in enumerate(robot_containers):
        container.net_wrapper = net_wrappers[i]
        args_field = container.robot.getField("controllerArgs")
        nn_string = json.dumps(net_wrappers[i].net.to_dict())

        #Set the new network for the argument.
        #If it doesn't have an arguments array yet, add one. Otherwise just modify it.
        if args_field.getCount() == 0:
            args_field.insertMFString(0, nn_string)
        else:
            args_field.setMFString(0, nn_string)

        container.robot.restartController()


def update_fitnesses(containers):
    for i, container in enumerate(containers):
        robot = container.robot
        end_coord = robot.getField("translation").getSFVec3f()
        container.net_wrapper.fitness = sum((end_coord[i] - container.start_coord[i])**2 for i in range(3))


def count_robot_motors(robot):
    def _count_field_motors(field):
        motor_count = 0
        for i in range(field.getCount()):
            child = field.getMFNode(i)
            if child.getType() == child.HINGE_JOINT:
                end_point = child.getField("endPoint")
                if end_point is not None:
                    motor_count += _count_field_motors(end_point.getSFNode().getField("children"))
                if child.getField("device").getCount() > 0:
                    motor_count += 1
        return motor_count

    if robot.isProto():
        children = robot.getProtoField("children")
    else:
        children = robot.getField("children")

    return _count_field_motors(children)


fitness_cycle_counter = 0
def fitness_function_callback(new_population, epoch_time):
    global fitness_cycle_counter
    prepare_cycle(containers, new_population)
    fitness_cycle_counter += 1
    print(f"Starting cycle {fitness_cycle_counter}.")
    run()

    start_time = supervisor.getTime()
    while supervisor.getTime() < (start_time + epoch_time):
        if supervisor.step(timestep) == -1:
            sys.exit()

    pause()
    update_fitnesses(containers)


if __name__ == "__main__":
    print("\x1b[2J")  # Clear the terminal.

    parser = argparse.ArgumentParser(description='Supervisor for evolving robot movement strategies.')
    parser.add_argument('--minweight', default=-2, type=int, help='Minimum weight for the neural network')
    parser.add_argument('--maxweight', default=2, type=int, help='Maximum weight for the neural network')
    parser.add_argument('--minbias', default=-2, type=int, help='Minimum bias for the neural network')
    parser.add_argument('--maxbias', default=2, type=int, help='Maximum bias for the neural network')
    parser.add_argument('--layercount', default=3, type=int, help='Number of layers for the neural network')
    parser.add_argument('--epoch', default=5, type=float, help='Epoch time in seconds for each fitness trial.')
    parser.add_argument('--maxmutationcount', default=20, type=int, help='Defines the maximum number of mutations possible in the generation of a new neural network')
    parser.add_argument('--seed', default=None, type=int, help='Defines the random seed')
    parser.add_argument('--outpath', default="out.csv", type=str, help='CSV output path')


    args = parser.parse_args()
    if args.minweight >= args.maxweight:
        print("Error! minweight must be less than maxweight")
        exit(1)

    if args.minbias >= args.maxbias:
        print("Error! minbias must be less than maxbias")
        exit(1)

    if args.layercount <= 0:
        print("Error! layer count must be greater than 0.")
        exit(1)

    if args.epoch <= 0:
        print("Error! epoch must be greater than 0.")
        exit(1)

    random.seed(args.seed)

    weight_min_max = (args.minweight, args.maxweight)
    bias_min_max = (args.minbias, args.maxbias)


    print("Starting supervisor")
    pause()
    containers = initialize_robots()

    if len(containers) < 2:
        print("Error! At least 2 robots must be defined!")
        exit(1)

    motor_count = count_robot_motors(containers[0].robot)
    new_generation_size = len(containers)
    nn_output_count = motor_count
    nn_input_count = nn_output_count + 1
    ea = EASupervisor(fitness_function_callback, new_generation_size, nn_input_count, nn_output_count, weight_min_max, bias_min_max, layer_count=args.layercount, output_path=args.outpath, max_mutation_count=args.maxmutationcount, callback_args=[args.epoch])
    ea.run()
