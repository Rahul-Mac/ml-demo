#!/usr/bin/env python3

# Demonstration of feed forward neural network using back propagation

__author__ = "Rahul Mac"

import numpy as np

class Neural_Network():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))
    
    def sigmoid_derivative(self, x):
        return(x*(1-x))

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return(output)

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

if __name__ == "__main__":
    nn = Neural_Network()
    print("Random starting weights: " + str(nn.synaptic_weights))
    i = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    o = np.array([[0, 1, 1, 0]]).T
    epoch = 50000
    nn.train(i, o, epoch) 
    print("Synaptic weights after training: " + str(nn.synaptic_weights))
    A = str(input("1st input"))
    B = str(input("2nd input"))
    C = str(input("3rd input"))

    print(nn.think(np.array([A, B, C])))
