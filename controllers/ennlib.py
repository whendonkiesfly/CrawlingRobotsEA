"""
    This library defines classes that implement neural networks as well as a
    generic EA algorithm.
"""

import json
import pickle
import random

import numpy as np

def interpolate(val, min, max):
    """
    Interpolates values between 0 and 1 to values between the specified min and max.
    Used primarily map values generated by random.random() to a specified range.
    """
    return val * (max - min) + min


class NNLayer:
    """
    Contains one layer of a neural network and handles the math for that layer.
    """
    def __init__(self, weight_matrix, bias_vector, weight_min_max, bias_min_max):
        self.weight_matrix = np.array(weight_matrix)
        self.bias_vector = np.array(bias_vector)
        self.weight_min_max = np.array(weight_min_max)
        self.bias_min_max = np.array(bias_min_max)

    def to_dict(self):
        """
        Converts the layer to a dictionary that can be serialized.
        """
        return {
            "weight_matrix": self.weight_matrix.tolist(),
            "bias_vector": self.bias_vector.tolist(),
            "weight_min_max": self.weight_min_max.tolist(),
            "bias_min_max": self.bias_min_max.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        """
        Converts a dictionary as produced by to_dict() back to a NNLayer object.
        """
        return cls(**d)

    def __str__(self):
        return f"{self.weight_matrix}\n{self.bias_vector}"


    @classmethod
    def random(cls, input_count, output_count, weight_min_max, bias_min_max):
        """
        Generates a randomized NNLayer object.
        weight_min_max and bias_min_max should be tuples or lists of length 2 where the first value is the min and the second is the max.
        """
        #Generate a layer with random biases and weight matrix.
        biases = interpolate(np.random.rand(output_count), *bias_min_max)
        weights = interpolate(np.random.rand(output_count, input_count), *weight_min_max)
        return cls(weights, biases, weight_min_max, bias_min_max)

    @property
    def input_count(self):
        return len(self.weight_matrix[0])

    @property
    def output_count(self):
        return len(self.weight_matrix)

    def process_layer(self, inputs):
        """
        Given inputs, returns the values calculated for this layer of the neural net.
        """
        assert len(inputs) == self.input_count, f"inputs must be of length {self.input_count} but got {len(inputs)}"
        raw_value = self.weight_matrix.dot(inputs) + self.bias_vector
        return self.activation_func(raw_value)

    def activation_func(self, values):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-values))

    def mutate(self):
        """
        Mutates the network layer by randomizing one weight or bias value.
        """
        #Use a random number to decide whether we should mutate a weight or bias.
        if random.random() <= 1/self.input_count:
            #Mutate a bias.
            self.bias_vector[random.randrange(len(self.bias_vector))] = interpolate(random.random(), *self.bias_min_max)
        else:
            #Mutate a weight.
            self.weight_matrix[random.randrange(len(self.weight_matrix))][random.randrange(len(self.weight_matrix[0]))] = interpolate(random.random(), *self.weight_min_max)

    def crossover(self, other_parent, crossover_ratio=0.1):
        """
        Crossover is done by selecting which parent to take each weight and bias from. A new NNLayer is returned with these values.
        Higher values for crossover_ratio will cause the child to get more traits from the other parent. This value can be between 0 and 1
        and 0.5 will give a 50% chance that either parent will be selected for each value.
        """
        assert 0 <= crossover_ratio <= 1, "crossover_ratio must be between 0 and 1."
        #Crossover biases.
        bias_vector = np.array([self.bias_vector[i] if random.random() > crossover_ratio else other_parent.bias_vector[i] \
                                                                                        for i in range(len(self.bias_vector))])
        #Crossover weights
        weight_matrix = np.array([[self.weight_matrix[i, j] if random.random() > crossover_ratio else other_parent.weight_matrix[i, j]\
                                                                                    for j in range(len(self.weight_matrix[0]))] for i in range(len(self.weight_matrix))])
        #Create the child.
        return type(self)(weight_matrix, bias_vector, self.weight_min_max, self.bias_min_max)



class BasicNeuralNet:
    """
    Stores an evolvable neural network with methods for processing it.
    """

    def __init__(self, layers):
        """
        layers is a list of NNLayer objects.
        """
        self.layers = layers

    def to_dict(self):
        """
        Creates a dictionary object that can be serialized.
        """
        return {
            "layers": [layer.to_dict() for layer in self.layers]
        }

    @classmethod
    def from_dict(cls, d):
        """
        Creates a BasicNeuralNet object from the dictionary produced by to_dict().
        """
        layers = [NNLayer.from_dict(layerdata) for layerdata in d["layers"]]
        return cls(layers=layers)

    @classmethod
    def random(cls, input_count, output_count, weight_min_max, bias_min_max, layer_count=3):
        """
        Generates a random neural network.
        input_count is the number of input nodes.
        output_count is the number of output nodes
        weight_min_max defines the minimum and maximum weights for the generated network. It must be in the format (min, max)
        bias_min_max defines the minimum and maximum biases for the generated network. It must be in the format (min, max)
        layer_count defines the number of layers in the network.
        """
        layers = [
            NNLayer.random(input_count, input_count, weight_min_max, bias_min_max) for _ in range(layer_count-1)
        ]

        layers.append(NNLayer.random(input_count, output_count, weight_min_max, bias_min_max))

        return cls(layers)

    @property
    def input_count(self):
        return self.layers[0].input_count

    @property
    def output_count(self):
        return self.layers[-1].output_count


    def process_network(self, inputs):
        """
        Processes the inputs through the network and returns the results.
        """
        output = inputs
        for layer in self.layers:
            output = layer.process_layer(output)
        return output

    def crossover(self, other_parent, crossover_ratio=0.1):
        """
        Creates a new neural network using crossover with another BasicNeuralNet object.
        See NNLayer.crossover for information on crossover_ratio.
        """
        child_layers = [self.layers[i].crossover(other_parent.layers[i]) for i in range(len(self.layers))]
        return type(self)(child_layers)

    def mutate(self):
        """
        Calls the mutate method on one randomly selected layer.
        """
        self.layers[random.randrange(len(self.layers))].mutate()


class NetWrapper:
    """
    Structure for storing a network and the associated fitness.
    """
    def __init__(self, net, fitness=None):
        self.net = net
        self.fitness = fitness


class EASupervisor:
    """
    Manages the EA algorithm for evolving neural networks.
    Uses a callback fitness function.
    """
    def __init__(self, fitness_callback, popcount, input_count, output_count, weight_min_max, bias_min_max, layer_count=3, max_mutation_count=5, output_path="out.csv", callback_args=None):
        """
        fitness_callback defines the callback function. The first parameter this function is
            passed is a list of NetWrapper objects. The fitness value for each of these must be set.
            All of the newly generated networks will be passed in during each cycle.
            Raising an exception or an exit() call will stop the EA.
        popcount sets the number of nodes in the population as well as the number of child nodes generated each cycle.
        input_count sets the number of input nodes to the networks.
        output_count sets the number of output nodes to the networks.
        weight_min_max defines the minimum and maximum weights for the generated networks. It must be in the format (min, max).
        bias_min_max defines the minimum and maximum biases for the generated networks. It must be in the format (min, max).
        layer_count defines the number of layers in the networks.
        max_mutation_count sets the maximum for the randomly selected number of mutations for each new child.
        output_path sets the file path for the csv output file.
        callback_args sets additional parameters to be passed to the fitness_callback function.
        """
        self.population = [NetWrapper(BasicNeuralNet.random(input_count, output_count, weight_min_max, bias_min_max, layer_count)) for i in range(popcount)]
        self.max_mutation_count = max_mutation_count
        self.popcount = popcount
        self.fitness_callback = fitness_callback
        self.callback_args = callback_args if callback_args is not None else []
        self.output_path = output_path
        self.best_fitness = -float("inf")

        #Clear out the file.
        with open(self.output_path, "w") as fout:
            fout.write("Best Fitness, Average Fitness, Best Network\n")
        self.fitness_callback(self.population, *self.callback_args)

    def run_cycle(self, crossover_ratio=0.5, dominance_exp=1.0):
        """
        Runs a single cycle of the EA.
        crossover_ratio and dominance_exp are parameters passed into the make_baby function.
        See the documentation there for information on those parameters.
        """
        #Birth new NNs
        new_babies = [NetWrapper(self.make_baby(crossover_ratio=crossover_ratio, dominance_exp=dominance_exp)) for _ in range(self.popcount)]

        #Run fitness functions.
        self.fitness_callback(new_babies, *self.callback_args)

        self.population.extend(new_babies)

        #Reduce the population size.
        self.reduce_pop()

        #Output the best fitness.
        best_fitness = self.population[0].fitness
        average_fitness = sum(robot.fitness for robot in self.population) / len(self.population)
        print(f"Best: {best_fitness}    Average: {average_fitness}")

        #If we have a new best network, we want to output it in the CSV file.
        if best_fitness > self.best_fitness:
            #We have a new best network. We want to output its json string in the csv file.
            self.best_fitness = best_fitness
            new_best_network = json.dumps(self.population[0].net.to_dict())
        else:
            #No new best. Make it an empty string.
            new_best_network = ""

        #Output data to the CSV file.
        with open(self.output_path, "a") as fout:
            fout.write(f"{best_fitness}, {average_fitness}, {new_best_network}\n")

    def run(self):
        """
        Runs the run_cycle function forever. Raising an exception or an exit() call will stop the EA.
        """
        while True:
            self.run_cycle()


    def make_baby(self, crossover_ratio=0.5, dominance_exp=1.0):
        """
        Selects a pair to mate and returns a new NN which has had crossover and mutations applied.
        Higher values for crossover_ratio determines the way the parameters of each parent are split.
            0.5 will give an equal probability of each parameter coming from each parent.
        dominance_exp is used to direct the parent selection process. Higher dominance_exp values will
            significantly reduce the probability of lower fitness networks being selected to mate.
            A dominance value of 1 will cause the probability of a network to be selected close to
            the ratio of its fitness to the sum of fitnesses in the population.
        """
        a = b = None
        counter = 0
        while a is b:
            if counter == 10:
                raise Exception("Failed to select two unique parents after 10 tries")
            counter += 1
            a, b = random.choices([item.net for item in self.population],
                                (item.fitness**dominance_exp for item in self.population), k=2)
        child = a.crossover(b, crossover_ratio)
        for _ in range(random.randrange(self.max_mutation_count)):
            child.mutate()

        return child

    def reduce_pop(self):
        """
        Sorts the population in decending order and cuts off the half with the lower fitnesses.
        """
        #Sort descending by the fitness
        self.population.sort(key=lambda item: item.fitness, reverse=True)
        #Delete the low fitness items.
        del self.population[self.popcount:]
