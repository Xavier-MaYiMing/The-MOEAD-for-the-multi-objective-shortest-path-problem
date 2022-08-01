#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/28 12:29
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : MOEAD4MOSPP.py
# @Statement :The MOEA/D for the multi-objective shortest path problem
# @Reference : Zhang Q, Li H. MOEA/D: A multiobjective evolutionary algorithm based on decomposition[J]. IEEE Transactions on Evolutionary Computation, 2007, 11(6): 712-731.
# @Reference : Ahn C W, Ramakrishna R S. A genetic algorithm for shortest path routing problem and the sizing of populations[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(6): 566-579.
import copy
import numpy as np
import random
import math


def find_neighbor(network):
    """
    find the neighbor of each node
    :param network:
    :return: {node 1: [the neighbor nodes of node 1], ...}
    """
    nn = len(network)
    neighbor = []
    for i in range(nn):
        neighbor.append(list(network[i].keys()))
    return neighbor


def random_path_generator(source, destination, neighbor):
    """
    generate random path
    :param source: source node
    :param destination: destination node
    :param neighbor: neighbor
    :return:
    """
    path = [source]
    while path[-1] != destination:
        temp_node = path[-1]
        neighbors = neighbor[temp_node]
        node_set = []
        for node in neighbors:
            if node not in path:
                node_set.append(node)
        if node_set:
            path.append(random.choice(node_set))
        else:
            path = [source]
    return path


def cal_obj(network, path, nw):
    """
    calculate the fitness of an individual
    :param network:
    :param path:
    :param nw:
    :return:
    """
    obj = [0 for i in range(nw)]
    for i in range(len(path) - 1):
        for j in range(nw):
            obj[j] += network[path[i]][path[i + 1]][j]
    return obj


def cal_fitness(network, population, nw):
    """
    calculate the fitness of a population
    :param network:
    :param population:
    :param nw:
    :return:
    """
    for item in population:
        item['objective'] = cal_obj(network, item['chromosome'], nw)
    return population


class Mean_vector:
    # 对m维空间，目标方向个数H
    def __init__(self, H=5, m=3):
        self.H = H
        self.m = m
        self.stepsize = 1 / H

    def perm(self, sequence):
        # ！！！ 序列全排列，且无重复
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_mean_vectors(self):
    #生成权均匀向量
        H = self.H
        m = self.m
        sequence = []
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        ws = []

        pe_seq = self.perm(sequence)
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)
        return ws


def search_neighbor(lambda_list, neighbor_size, pop):
    """
    calculate the set of nearest individuals for each individual
    :param lambda_list:
    :param neighbor_size: the number of neighbors
    :param pop:
    :return:
    """
    B = []
    distance = []
    for i in range(pop):
        distance.append([0 for n in range(pop)])
    for i in range(pop):
        for j in range(i, pop):
            lambda1 = lambda_list[0]
            lambda2 = lambda_list[1]
            dist = 0
            for k in range(len(lambda1)):
                dist += (lambda2[k] - lambda1[k]) ** 2
            dist = math.sqrt(dist)
            distance[i][j] = dist
            distance[j][i] = dist
        temp_list = np.array(distance[i])
        index = np.argsort(temp_list)
        index = index.tolist()
        B.append(index[: neighbor_size])
    return B


def pareto_dominated(obj1, obj2):
    """
    judge whether individual 1 is Pareto dominated by individual 2
    :param obj1: the objective of individual 1
    :param obj2: the objective of individual 2
    :return:
    """
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] < obj2[i]:
            return False
        elif obj1[i] > obj2[i]:
            sum_less += 1
    if sum_less != 0:
        return True
    return False


def non_domination_sort(population):
    """
    non domination sort
    :param population:
    :return:
    """
    pop = len(population)
    index = 1
    pareto_rank = {index: []}
    for i in range(pop):
        population[i]['n'] = 0  # domination counter
        population[i]['s'] = []  # the set of solutions dominated by population[i]
        temp_obj = population[i]['objective']
        for j in range(pop):
            if i != j:
                temp_population = population[j]
                if pareto_dominated(temp_obj, temp_population['objective']):
                    population[i]['n'] += 1
                elif pareto_dominated(temp_population['objective'], temp_obj):
                    population[i]['s'].append(j)
        if population[i]['n'] == 0:
            pareto_rank[index].append(i)
            population[i]['pareto rank'] = index
    while pareto_rank[index]:
        pareto_rank[index + 1] = []
        q_index = index + 1
        for p in pareto_rank[index]:
            for q in population[p]['s']:
                population[q]['n'] -= 1
                if population[q]['n'] == 0:
                    pareto_rank[q_index].append(q)
                    population[q]['pareto rank'] = q_index
        index += 1
    return population


def crossover(chromosome1, chromosome2):
    """
    the crossover operation of two individuals
    :param chromosome1:
    :param chromosome2:
    :return:
    """
    potential_crossing_site = []
    for i in range(1, len(chromosome1) - 1):
        for j in range(1, len(chromosome2) - 1):
            if chromosome1[i] == chromosome2[j]:
                potential_crossing_site.append([i, j])
    if potential_crossing_site:
        crossing_site = random.choice(potential_crossing_site)
        offspring1 = chromosome1[0: crossing_site[0]]
        offspring1.extend(chromosome2[crossing_site[1]:])
        return offspring1
    else:
        return chromosome1


def mutation(chromosome, destination, neighbor):
    """
    the mutation operation of an individual
    :param chromosome:
    :param destination:
    :param neighbor:
    :return:
    """
    temp_index = random.randint(1, len(chromosome) - 1)
    new_chromosome = chromosome[: temp_index]
    while True:
        temp_node = new_chromosome[-1]
        if temp_node == destination:
            break
        neighbors = neighbor[temp_node]
        node_set = []
        for node in neighbors:
            if node not in new_chromosome:
                node_set.append(node)
        if node_set:
            new_chromosome.append(random.choice(node_set))
        else:
            temp_index = random.randint(1, len(chromosome) - 1)
            new_chromosome = chromosome[: temp_index]
    return new_chromosome


def find_EP(population):
    """
    select the EP from the population
    :param population:
    :return:
    """
    EP = []
    ep_path = []
    for i in range(len(population)):
        if population[i]['pareto rank'] == 1 and population[i]['chromosome'] not in ep_path:
            ep_path.append(population[i]['chromosome'])
            temp_population = copy.deepcopy(population[i])
            EP.append(temp_population)
    return EP, ep_path


def tchebycheff_approach(obj, z, lambda_value):
    """

    :param obj:
    :param z:
    :param lambda_value:
    :return:
    """
    max_value = 0
    for i in range(len(z)):
        temp_value = lambda_value[i] * abs(z[i] - obj[i])
        max_value = max(max_value, temp_value)
    return max_value


def update_neighbor(population, child, B, i, child_obj, z):
    """
    update of neighboring solutions
    :param population:
    :param child:
    :param B:
    :param i:
    :param child_obj:
    :param z:
    :return:
    """
    flag = False
    index_list = B[i]
    for iter in range(len(index_list)):
        temp_index = index_list[iter]
        item = population[temp_index]
        temp_Tchebycheff = tchebycheff_approach(child_obj, z, item['lambda'])
        if temp_Tchebycheff <= item['Tchebycheff']:
            flag = True
            population[temp_index] = {
                'chromosome': child,
                'objective': child_obj,
                'Tchebycheff': temp_Tchebycheff,
                'lambda': item['lambda'],
            }
    return population, flag


def main(network, source, destination, h):
    """
    the main function
    :param network: {node 1: {node 2: [weight1, weight2, ...], ...}, ...}
    :param source: the source node
    :param destination: the destination node
    :param h: the uniformly distributed number on each objective
    :return:
    """
    gen = 100  # the maximum number of generations (iterations)
    p_mutation = 0.15  # mutation probability
    inf = 10e6
    neighbor = find_neighbor(network)
    nw = len(network[source][neighbor[source][0]])  # the number of objectives
    population = []
    mv = Mean_vector(h, nw)
    lambda_list = mv.get_mean_vectors()
    pop = len(lambda_list)
    neighbor_size = 20  # neighbor size
    neighbor_index = [i for i in range(neighbor_size)]
    for i in range(pop):
        temp_path = random_path_generator(source, destination, neighbor)
        population.append({
            'chromosome': temp_path
        })
    population = cal_fitness(network, population, nw)

    # Initialize the reference point
    z = []  # reference point
    for i in range(nw):
        z.append(inf)
    for item in population:
        for i in range(nw):
            z[i] = min(z[i], item['objective'][i])

    # Add the Tchebycheff distance
    for i in range(pop):
        item = population[i]
        item['lambda'] = lambda_list[i]
        item['Tchebycheff'] = tchebycheff_approach(item['objective'], z, lambda_list[i])

    # Initialize the neighbor size
    B = search_neighbor(lambda_list, neighbor_size, pop)

    # Initialize an external population (EP) to store non-dominated solutions
    population = non_domination_sort(population)
    EP, ep_path = find_EP(population)

    # The main loop
    for iteration in range(gen):
        for i in range(pop):
            [index1, index2] = random.sample(neighbor_index, 2)
            parent1 = population[B[i][index1]]['chromosome']
            parent2 = population[B[i][index2]]['chromosome']
            child = crossover(parent1, parent2)
            if random.random() < p_mutation:
                child = mutation(child, destination, neighbor)
            child_obj = cal_obj(network, child, nw)

            # Update reference point
            flag = False
            for j in range(nw):
                if z[j] > child_obj[j]:
                    flag = True
                    z[j] = child_obj[j]
            if flag:
                for j in range(pop):
                    item = population[j]
                    item['Tchebycheff'] = tchebycheff_approach(item['objective'], z, item['lambda'])

            # Update neighborhood solutions
            population, flag = update_neighbor(population, child, B, i, child_obj, z)

            # Update EP
            if flag and child not in ep_path:
                flag1 = True
                need_to_remove = []
                for ind in range(len(EP)):
                    item = EP[ind]
                    item_obj = item['objective']
                    if pareto_dominated(item_obj, child_obj):
                        need_to_remove.append(ind)
                    if flag1 and pareto_dominated(child_obj, item_obj):
                        flag1 = False
                for item in need_to_remove:
                    EP.remove(item)
                if flag1:
                    ep_path.append(child)
                    EP.append({
                        'chromosome': child,
                        'objective': child_obj,
                        'Tchebycheff': tchebycheff_approach(child_obj, z, lambda_list[i]),
                    })

    # Sort the results
    result = []
    for item in EP:
        result.append({
            'path': item['chromosome'],
            'objective': item['objective'],
        })
    return result


if __name__ == '__main__':
    test_network = {
        0: {1: [62, 50], 2: [44, 90], 3: [67, 10]},
        1: {0: [62, 50], 2: [33, 25], 4: [52, 90]},
        2: {0: [44, 90], 1: [33, 25], 3: [32, 10], 4: [52, 40]},
        3: {0: [67, 10], 2: [32, 10], 4: [54, 100]},
        4: {1: [52, 90], 2: [52, 40], 3: [54, 100]},
    }
    source_node = 0
    destination_node = 4
    print(main(test_network, source_node, destination_node, 20))
