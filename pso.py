#!/usr/bin/env python3

__author__ = "Rahul Mac"

import random as rd
from re import search
import numpy as np
from matplotlib import pyplot
import os

W = 0.5
c1 = 0.8
c2 = 0.9

n_iterations = 50
n_particles = 30
target_error = 1e-6

class Particle():
    def __init__(self):
        x = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        y = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        self.position = np.array([x, y])
        self.pBest_position = self.position
        self.pBest_value = float('inf')
        self.velocity = np.array([0, 0])

    def update(self):
        self.position += self.velocity

class Space():
    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gBest_value = float('inf')
        self.gBest_position = np.array([rd.random() * 50, rd.random() * 50])

    def fitness(self, particle):
        x = particle.position[0]
        y = particle.position[1]
        f = x**2 + y**2 + 1
        return f

    def set_pBest(self):
        for particle in self.particles:
            fitness_candidate = self.fitness(particle)
            if particle.pBest_value > fitness_candidate:
                particle.pBest_value = fitness_candidate
                particle.pBest_position = particle.position

    def set_gBest(self): 
        for particle in self.particles:
            best_fitness_candidate = self.fitness(particle)
            if self.gBest_value > best_fitness_candidate:
                self.gBest_value = best_fitness_candidate
                self.gBest_position = particle.position

    def update_particle(self):
        for particle in self.particles:
            global W
            inertial = W * particle.velocity
            self_confidence = c1 * rd.random() * (particle.pBest_position - particle.position)
            swarm_confidence = c2 * rd.random() * (self.gBest_position - particle.position)
            new_velocity = inertial + self_confidence + swarm_confidence
            particle.velocity = new_velocity
            particle.update()

    def show_particles(self, iteration):
        output_path = "./output"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        print(iteration, 'iteration')
        print("Best position in this time: ", self.gBest_position)
        print("Best value in this time: ", self.gBest_value)

        for particle in self.particles:
            pyplot.plot(particle.position[0], particle.position[1], 'ro')
            pyplot.plot(self.gBest_position[0], self.gBest_position[1], 'bo')
        pyplot.savefig('output/image'+str(iteration)+'.png')


search_space = Space(1, target_error, n_particles)
particle_vector = [Particle() for _ in range(search_space.n_particles)]
search_space.particles = particle_vector

iteration = 0
while(iteration < n_iterations):
    search_space.set_pBest()
    search_space.set_gBest()

    search_space.show_particles(iteration)
    if(abs(search_space.gBest_value - search_space.target) <= search_space.target_error):
        break

    search_space.update_particle()
    iteration += 1
print("The best solution is: ", search_space.gBest_position, " in ", " iteration ")
