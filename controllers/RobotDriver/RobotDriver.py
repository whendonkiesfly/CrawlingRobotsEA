"""RobotDriver controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

import os
import sys
import json

import controller

#Set up the import path for ennlib.
sys.path.insert(0, os.path.join(os.path.split(os.path.realpath(__file__))[0], ".."))
from ennlib import BasicNeuralNet



def scale(val, in_range, out_range):
    """
    Scales a value between one range and another.
    in_range should be of format (in_min, in_max)
    out_range should be of format (out_min, out_max)
    """
    ret = (val-in_range[0])/(in_range[1]-in_range[0])*(out_range[1]-out_range[0])+out_range[0]
    return ret


class MotorInfo:
    """
    Structure for storing a reference to a motor on a robot and the type.
    """
    POSITION = 1
    VELOCITY = 2

    def __init__(self, motor):
        """
        Stores the motor and type.
        """
        self.motor = motor
        pos_range = (motor.getMinPosition(), motor.getMaxPosition())
        #Webots indicates a motor has infinite rotation by setting its min and max position to 0.
        if pos_range[0] == 0 and pos_range[1] == 0:
            #This is an infinite rotation motor. We want to control it using velocity rather than position.
            self.mode = self.VELOCITY
            #In order to set a motor to velocity mode, Webots requires us to set its position to infinity.
            motor.setPosition(float("inf"))
            #The range of allowed values is negative to positive max speed.
            #This speed is in rads/sec for rotational motors and meters/sec for linear motors.
            max_v = motor.getMaxVelocity()
            range = (-max_v, max_v)
        else:
            #This is not an infinite rotation motor. Control it with position.
            self.mode = self.POSITION
            range = pos_range
        #There can be issues with going right up to the limits (floating point error?), so we make the range slightly smaller.
        self.range = (scale(0.001, (0,1), range), scale(0.999, (0,1), range))



def set_joint_values(motor_infos, joint_values):
    """
    Sets motor values for both velocity controlled and position controlled joints.
    MotorInfo is a list of MotorInfo objects.
    joint_values should be a list of values between 0 and 1 where there is one joint value per motor.
    """
    assert len(motor_infos) == len(joint_values), f"Invalid joint count: Expected {len(motor_infos)} but got {len(joint_values)}."

    #Iterate over each motor and joint value.
    for motor_info, joint_value in zip(motor_infos, joint_values):
        #Scale the value to the appropriate range.
        joint_value = scale(joint_value, (0, 1), motor_info.range)

        #Depending on the type of motor, set the value differently.
        if motor_info.mode == MotorInfo.POSITION:
            #The motor is position controlled. Set position.
            motor_info.motor.setPosition(joint_value)

        elif motor_info.mode == MotorInfo.VELOCITY:
            #The motor is velocity controlled. Set the velocity.
            motor_info.motor.setVelocity(joint_value)

        else:
            #Something went wrong.
            raise Exception("Invalid motor mode:", motor_info.mode)

def get_joint_values(motors):
    #Gets the values from the motor (either position or velocity) and scales it to a value between 0 and 1.
    get_val = lambda motor_info: motor_info.motor.getTargetPosition() if motor_info.mode is MotorInfo.POSITION else motor_info.motor.getVelocity()
    return [scale(get_val(motor_info), motor_info.range, (0, 1)) for motor_info in motors]


#Bring in the arguments.
if len(sys.argv) < 2 or sys.argv[1] == "":
    print("No network file received in parameter.")
    sys.exit(1)
else:
    try:
        with open(sys.argv[1], "r") as fin:
            nn_string = fin.read()
            net = BasicNeuralNet.from_dict(json.loads(nn_string))
    except Exception as e:
        print("Error! Failed to read NN string.", e)
        exit(1)

#Create the Robot instance.
robot = controller.Robot()

#Get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

#Gather the motors.
motors = []
for i in range(robot.getNumberOfDevices()):
    device = robot.getDeviceByIndex(i)
    if isinstance(device, controller.Motor):
        motors.append(MotorInfo(device))


#In addition to sending the motor values as inputs to the neural network, a timer value
#is passed through as well in hopes of encouraging the network to have a cyclic pattern.
#This signal is a triangle wave and these are the parameters.
MOVE_PERIOD_TIME_SECS = 1  # Defines the period of the clock value signal to the NN.
CLOCK_VAL_AMPLITUDE = 1  # Defines the maximum value for the clock signal to the NN.

#Main loop
while robot.step(timestep) != -1:
    #Calculate the timer value.
    timer_input = (robot.getTime() % MOVE_PERIOD_TIME_SECS) * CLOCK_VAL_AMPLITUDE

    #Get the current state of the joints.
    joint_inputs = get_joint_values(motors)

    #Send the inputs through the neural network and set joints using its outputs.
    inputs = [timer_input] + joint_inputs
    set_joint_values(motors, net.process_network(inputs))
