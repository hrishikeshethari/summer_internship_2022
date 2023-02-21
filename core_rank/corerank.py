import numpy as np
import operator
import networkx as nx
import matplotlib.pyplot as plt
import pylab
from collections import defaultdict
import test 


def core_rank(edges, good_nodes, beta = 0.85, MAX_ITER = 100):
    maxer = 10
    nodes_dict = defaultdict(set)
    #edges = test.generate_random_graph(10)
    for edge in edges:
        nodes_dict[edge[0]].add(edge[1])
        nodes_dict[edge[1]].add(edge[0])
    #print(nodes_dict)

    # M is the Transition Matrix
    # v is the matrix that defines the probability of the random surfer of being at any paricular node
    M = np.zeros((maxer+1, maxer+1))
    v = np.zeros(maxer + 1)

    # Defining the Transition matrix
    for from_node in nodes_dict:
        length = len(nodes_dict[from_node])
        fraction = 1/length
        for to_node in nodes_dict[from_node]:
            M[to_node][from_node] = fraction

    # Defining initial v matrix
    no_of_nodes = len(test.flatten_and_unique(edges))
    fraction = 1 / no_of_nodes
    for i in range(1, maxer + 1):
        if i in good_nodes:
            v[i] = fraction
        else:
            v[i] = 0

    # Definining the teleport matrix which takes care of the Dead ends and Spider traps
    teleport = (1 - beta) * v

    M = beta * M
    count = 0
    # Carrying out the iterations until matrix v stops changing
    while(count < MAX_ITER):
        v1 = np.dot(M, v) + teleport
        v = v1
        count += 1
        
    #print("No. of iterations required without considering TrustRank: " + str(count))


    # Sorting nodes with respect to the final ranks of the nodes
    page_rank_score = []
    for i in range(1, len(v)):
        if v[i] != 0:
            page_rank_score.append([i, v[i]])

    sorted_core_rank_score = sorted(page_rank_score, key = operator.itemgetter(1), reverse = True)

    for idx, ele in sorted_core_rank_score:
        print(f'{idx} --> {ele}')