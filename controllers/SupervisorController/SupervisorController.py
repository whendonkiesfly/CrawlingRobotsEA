"""
SupervisorController controller.

One robot with the supervisor bit must be set and running this controller in order to
execute the EA on the other robots.
"""


import argparse
import json
import os
import random
import string
import sys
import time
import tempfile

from controller import Supervisor
import controller

#Add to the import path in order to reference ennlib.
sys.path.insert(0, os.path.join(os.path.split(os.path.realpath(__file__))[0], ".."))
from ennlib import EASupervisor

#The neural networks are passed to the robots as a file because the command line
#arguments can be too big otherwise. These are placed into temporary files. This
#is used to make sure we don't have multiple temp files have name collisions.
tmpfilename_header = "".join(random.choice(string.ascii_uppercase) for _ in range(10))

class RobotContainer:
    """
    Container for storing a robot, its neural network, and the start pose of the robot.
    """
    def __init__(self, robot, start_coord, net_wrapper=None):
        self.robot = robot
        self.start_coord = start_coord
        self.net_wrapper = net_wrapper


def initialize_robots():
    """
    Initializes a RobotContainer object for each robot found in the environment and returns them in a list.
    """
    global supervisor

    robot_containers = []
    root_children = supervisor.getRoot().getField("children")
    #Iterate through all of the top level things in the environment looking for robots.
    for i in range(root_children.getCount()):
        node = root_children.getMFNode(i)
        #Check the type of node we found to see if it is a robot of interest (a robot and not a supervisor)
        if node.getType() == node.ROBOT and not node.getField("supervisor").getSFBool():
            #We found a robot.
            start_coord = node.getField("translation").getSFVec3f()
            robot_containers.append(RobotContainer(node, start_coord))
    return robot_containers


def pause():
    """
    Pauses the simulation
    """
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)


def run():
    """
    Puts the simulation in run mode.
    """
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_RUN)

cycle_count = 0
def prepare_cycle(robot_containers, net_wrappers):
    """
    Assigns neural networks to each robot.
    robot_containers must be a list of RobotContainer objects as created by initialize_robots().
    net_wrappers is the list of NetWrapper objects passed into the EA's fitness function callback.
    The number of robots and networks must be equal.
    Returns a list of temporary file paths which should be cleaned up later.
    """
    global tmpfilename_header
    global cycle_count
    assert len(robot_containers) == len(net_wrappers), f"Number of robots and networks must match. {len(robot_containers)}!={len(net_wrappers)}"
    filepaths = []
    for i, container in enumerate(robot_containers):
        container.net_wrapper = net_wrappers[i]
        args_field = container.robot.getField("controllerArgs")

        #Convert the network to json for transmission.
        nn_string = json.dumps(net_wrappers[i].net.to_dict())

        #Create the file that holds the NN
        nn_file_path = os.path.join(tempfile.gettempdir(), f"CrawlingEA_{tmpfilename_header}_{cycle_count}_{i}")
        filepaths.append(nn_file_path)
        with open(nn_file_path, "w") as fout:
            fout.write(nn_string)

        #Set the new network for the argument.
        #If it doesn't have an arguments array yet, add one. Otherwise just modify it.
        if args_field.getCount() == 0:
            args_field.insertMFString(0, nn_file_path)
        else:
            args_field.setMFString(0, nn_file_path)

        container.robot.restartController()

    cycle_count += 1

    #Restart the simulation to put everything back in the original position.
    supervisor.simulationReset()

    #Start the simulation.
    run()

    #Take a couple steps for the simulation reset to get processed and for the
    # system clock to regain its sanity.
    supervisor.step(timestep)
    supervisor.step(timestep)

    return filepaths


def update_fitnesses(containers):
    """
    Calculates the fitness of each robot and updates the fitness within the container after pausing the simulation.
    Fitness is equal to the euclidian distance from the robot's starting point to its current position.
    containers is the list of RobotContainer objects with the networks initialized in them.
    Euclidian distance is calculated without accounting for depth to reduce the likelyhood
    of a robot falling causing them to gain an advantage.
    """
    #Pause the simulation.
    pause()

    for i, container in enumerate(containers):
        robot = container.robot
        end_coord = robot.getField("translation").getSFVec3f()
        #Occationally the physics engine has issues and robots fall through the floor.
        #Throw these out by giving them a bad fitness.
        if end_coord[1] < 0:
            #Fell through the floor.
            container.net_wrapper.fitness = -float("inf")
        else:
            #Only look at indices 0 and 2. 1 is y (height) and we don't want to account for that.
            container.net_wrapper.fitness = sum((end_coord[i] - container.start_coord[i])**2 for i in (0, 2))


def count_robot_motors(robot):
    """
    Traverses a robot tree to count the number of motors.
    This works for robots whether or not they are defined by a proto file.
    """
    #TODO: Consider adding support for linear actuators.
    def _count_field_motors(field):
        motor_count = 0
        #Iterate through the field's nodes looking for a hinge joint.
        for i in range(field.getCount()):
            child = field.getMFNode(i)
            #If the node we found is a hinge joint, check it for a motor after recursing into its end point.
            if child.getType() == child.HINGE_JOINT:
                end_point = child.getField("endPoint")
                #If the joint has an end point, recurse into it.
                if end_point is not None:
                    motor_count += _count_field_motors(end_point.getSFNode().getField("children"))
                #Check to see if we have a motor.
                if child.getField("device").getCount() > 0:
                    motor_count += 1
        return motor_count

    #The method to get the robot fields depends on whether or not it was defined by a proto file.
    if robot.isProto():
        children = robot.getProtoField("children")
    else:
        children = robot.getField("children")

    return _count_field_motors(children)


fitness_cycle_counter = 0
def fitness_function_callback(new_population, epoch_time):
    """
    Fitness function for the EA.
    This assigns neural networks to the existing robots, runs for the specified
    amount of time, and then calculates and updates fitnesses.
    epoch_time defines the amount of time for the EA to run in seconds.
    """
    global fitness_cycle_counter

    #Assign neural networks to robots.
    tmp_files = prepare_cycle(containers, new_population)
    fitness_cycle_counter += 1
    print(f"Starting cycle {fitness_cycle_counter}.")
    try:

        #Wait for a while to let the robots move.
        start_time = supervisor.getTime()
        while supervisor.getTime() < (start_time + epoch_time):
            if supervisor.step(timestep) == -1:
                sys.exit()

        #Calculate the fitnesses.
        update_fitnesses(containers)

    finally:
        #Clean up temporary files.
        for tmp_file in tmp_files:
            os.remove(tmp_file)




#Create the Robot instance.
supervisor = Supervisor()

#Get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supervisor for evolving robot movement strategies.')
    parser.add_argument('--minweight', default=-2, type=int, help='Minimum weight for the neural network')
    parser.add_argument('--maxweight', default=2, type=int, help='Maximum weight for the neural network')
    parser.add_argument('--minbias', default=-2, type=int, help='Minimum bias for the neural network')
    parser.add_argument('--maxbias', default=2, type=int, help='Maximum bias for the neural network')
    parser.add_argument('--layercount', default=3, type=int, help='Number of layers for the neural network')
    parser.add_argument('--epoch', default=5, type=float, help='Epoch time in seconds for each fitness trial.')
    parser.add_argument('--maxmutationcount', default=20, type=int, help='Defines the maximum number of mutations possible in the generation of a new neural network')
    parser.add_argument('--seed', default=None, type=int, help='Defines the random seed')
    parser.add_argument('--motor_count_override', default=None, type=int, help='For some robots, the code is unable to count the motors. In this case, it must be explicitly specified.')
    parser.add_argument('--outpath', default="out.csv", type=str, help='CSV output path')


    args = parser.parse_args()
    if args.minweight >= args.maxweight:
        print("Error! minweight must be less than maxweight")
        parser.print_help()
        exit(1)

    if args.minbias >= args.maxbias:
        print("Error! minbias must be less than maxbias")
        parser.print_help()
        exit(1)

    if args.layercount <= 0:
        print("Error! layer count must be greater than 0.")
        parser.print_help()
        exit(1)

    if args.epoch <= 0:
        print("Error! epoch must be greater than 0.")
        parser.print_help()
        exit(1)

    random.seed(args.seed)

    weight_min_max = (args.minweight, args.maxweight)
    bias_min_max = (args.minbias, args.maxbias)


    print("Starting supervisor")

    #Pause the simulation
    pause()

    #Gather the robots
    containers = initialize_robots()

    if len(containers) < 2:
        print("Error! At least 2 robots must be defined!")
        parser.print_help()
        exit(1)

    #Count the motors so we know how many inputs and outputs we need.
    if args.motor_count_override is None:
        motor_count = count_robot_motors(containers[0].robot)
        if motor_count < 1:
            print("Error! No motors found on robots.")
            parser.print_help()
            exit(1)
    else:
        motor_count = args.motor_count_override

    new_generation_size = len(containers)
    nn_output_count = motor_count
    nn_input_count = nn_output_count + 1

    #Create and run the EA supervisor.
    ea = EASupervisor(fitness_function_callback, new_generation_size, nn_input_count, nn_output_count, weight_min_max, bias_min_max, layer_count=args.layercount, output_path=args.outpath, max_mutation_count=args.maxmutationcount, callback_args=[args.epoch])
    ea.run()
