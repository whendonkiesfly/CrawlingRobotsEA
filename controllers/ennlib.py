import json
import pickle
import random

import numpy as np

def interpolate(val, min, max):
    return val * (max - min) + min


def nn_encode(net):
    return b64(pickle.dumps(net)).decode('utf8')

def nn_decode(net_str):
    return pickle.loads(base64.b64decode(blah.encode('utf8')))


class NNLayer:
    def __init__(self, weight_matrix, bias_vector, weight_min_max, bias_min_max):
        self.weight_matrix = np.array(weight_matrix)
        self.bias_vector = np.array(bias_vector)
        self.weight_min_max = np.array(weight_min_max)
        self.bias_min_max = np.array(bias_min_max)

    def to_dict(self):
        return {
            "weight_matrix": self.weight_matrix.tolist(),
            "bias_vector": self.bias_vector.tolist(),
            "weight_min_max": self.weight_min_max.tolist(),
            "bias_min_max": self.bias_min_max.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def __str__(self):
        return f"{self.weight_matrix}\n{self.bias_vector}"


    @classmethod
    def random(cls, input_count, output_count, weight_min_max, bias_min_max):
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
        assert len(inputs) == self.input_count, f"inputs must be of length {self.input_count}"
        raw_value = self.weight_matrix.dot(inputs) + self.bias_vector
        return self.activation_func(raw_value)

    def activation_func(self, values):
        return 1 / (1 + np.exp(-values))

    def mutate(self):
        if random.random() <= 1/self.input_count:
            #Mutate a bias.
            self.bias_vector[random.randrange(len(self.bias_vector))] = interpolate(random.random(), *self.bias_min_max)
        else:
            #Mutate a weight.
            self.weight_matrix[random.randrange(len(self.weight_matrix))][random.randrange(len(self.weight_matrix[0]))] = interpolate(random.random(), *self.weight_min_max)

    def crossover(self, other_parent, crossover_ratio=0.1):
        #Crossover biases.
        bias_vector = np.array([self.bias_vector[i] if random.random() > crossover_ratio else other_parent.bias_vector[i] \
                                                                                        for i in range(len(self.bias_vector))])
        #Crossover weights
        weight_matrix = np.array([[self.weight_matrix[i, j] if random.random() > crossover_ratio else other_parent.weight_matrix[i, j]\
                                                                                    for j in range(len(self.weight_matrix[0]))] for i in range(len(self.weight_matrix))])
        return type(self)(weight_matrix, bias_vector, self.weight_min_max, self.bias_min_max)



class BasicNeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def to_dict(self):
        return {
            "layers": [layer.to_dict() for layer in self.layers]
        }

    @classmethod
    def from_dict(cls, d):
        layers = [NNLayer.from_dict(layerdata) for layerdata in d["layers"]]
        return cls(layers=layers)

    @classmethod
    def random(cls, input_count, output_count, weight_min_max, bias_min_max, layer_count=3):
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
        output = inputs
        for layer in self.layers:
            output = layer.process_layer(output)
        return output

    def crossover(self, other_parent, crossover_ratio=0.1):
        child_layers = [self.layers[i].crossover(other_parent.layers[i]) for i in range(len(self.layers))]
        return type(self)(child_layers)

    def mutate(self):
        self.layers[random.randrange(len(self.layers))].mutate()


class NetWrapper:
    def __init__(self, net, fitness=None):
        self.net = net
        self.fitness = fitness

class EASupervisor:
    def __init__(self, fitness_callback, popcount, input_count, output_count, weight_min_max, bias_min_max, layer_count=3, max_mutation_count=5, output_path="out.csv", callback_args=None):
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
        print(best_fitness, len(self.population))

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            new_best_network = json.dumps(self.population[0].net.to_dict())
        else:
            new_best_network = ""

        with open(self.output_path, "a") as fout:
            fout.write(f"{best_fitness}, {average_fitness}, {new_best_network}\n")

    def run(self):
        while True:
            self.run_cycle()


    def make_baby(self, crossover_ratio=0.5, dominance_exp=1.0):
        """
        Selects a pair to mate and returns a new NN which has had crossover and mutations applied.
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
        #Sort descending by the fitness
        self.population.sort(key=lambda item: item.fitness, reverse=True)
        #Delete the low fitness items.
        del self.population[self.popcount:]
