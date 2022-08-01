### The MOEA/D for the Multi-Objective Shortest Path Problem

----

##### Reference: Zhang Q, Li H. MOEA/D: A multiobjective evolutionary algorithm based on decomposition[J]. IEEE Transactions on Evolutionary Computation, 2007, 11(6): 712-731.

##### Reference: Ahn C W, Ramakrishna R S. A genetic algorithm for shortest path routing problem and the sizing of populations[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(6): 566-579.

----

The multi-objective aims to find a set of paths with minimized costs. 

| Variables     | Meaning                                                      |
| ------------- | ------------------------------------------------------------ |
| network       | Dictionary, {node 1: {node 2: [weight 1, weight 2, ...], ...}, ...} |
| source        | The source node                                              |
| destination   | The destination node                                         |
| h             | The uniformly distributed number on each objective           |
| lambda_list   | List, the uniformly distributed vectors                      |
| gen           | The maximum number of generations (iterations)               |
| pop           | Population size = len(lambda_list)                           |
| neighbor_size | Neighbor size                                                |
| B             | List, B[i] stores the indexes of the neighbor_size nearest neighbors of population[i] |
| p_mutation    | The probability of mutation                                  |
| neighbor      | List, [[the neighbor nodes of node 1], [the neighbor nodes of node 2], ...] |
| nw            | The number of objectives                                     |
| population    | List, all individuals ({'pareto rank': the Pareto rank (integer), 'chromosome': the chromosome (path), 'objective': the objective value on each objective (list)}) |
| z             | List, reference point, i.e., the best ever value of each objective |
| EP            | List, external population to store non-dominated solutions   |
| ep_path       | List, a set to record all non-dominated paths that have been found |

----

#### Example

![](https://github.com/Xavier-MaYiMing/The-MOEAD-for-the-multi-objective-shortest-path-problem/blob/main/MOSPP.png)

```python
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
```

##### Output:

```python
[
    {'path': [0, 3, 4], 'objective': [121, 110]}, 
    {'path': [0, 2, 4], 'objective': [96, 130]}, 
    {'path': [0, 3, 2, 4], 'objective': [151, 60]},
]
```

