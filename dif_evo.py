#!/usr/bin/env python3

__author__ = "Rahul Mac"

from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip, argmin, min, around
from matplotlib import pyplot

def obj(x):
    return x[0]**2.0 + x[1]**2.0

def mutation(x, F):
    return x[0] + F * (x[1] - x[2])

def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound

def crossover(mutated, target, dims, cr):
    p = rand(dims)
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial

def differential_evolution(pop_size, bounds, iter, F, cr):
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    obj_all = [obj(ind) for ind in pop]

    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj

    obj_iter = list()

    for i in range(iter):
        for j in range(pop_size):
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace = False)]

            mutated = mutation([a, b, c], F)
            mutated = check_bounds(mutated, bounds)
            trial = crossover(mutated, pop[j], len(bounds), cr)
            obj_target = obj(pop[j])
            obj_trial = obj(trial)

            if obj_trial < obj_target:
                pop[j] = trial
                obj_all[j] = obj_trial

        best_obj = min(obj_all)
        
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            obj_iter.append(best_obj)
            print('iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals = 5), best_obj))

    return [best_vector, best_obj, obj_iter]

pop_size = 10
bounds = asarray([(-5.0, 5.0), (-5.0, 5.0)])
iter = 100
F = 0.5
cr = 0.7

solution = differential_evolution(pop_size, bounds, iter, F, cr)
print('Solution: f([%s]) = %.5f' % (around(solution[0], decimals = 5), solution[1]))

pyplot.plot(solution[2], '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaulations f(x)')
pyplot.show()
