import networkx as nx
import numpy as np 
import math 
import operator
import matplotlib.pyplot as plt 
import corerank
import core_rank
from pagerank import PageRank

def flatten_and_unique(l):
    return set([item for sublist in l for item in sublist])

def generate_random_graph(number_of_nodes):
    import random 
    nodes = [_ for _ in range(10)]
        
    edges = [(random.choice(nodes), random.choice(nodes)) for _ in range(number_of_nodes)]
    unique_edges = []
    for edge in edges:
        if edge[0] != edge[1]:
            unique_edges.append(edge)

    return unique_edges

def main():
    unique_edges = [(3, 5), (1, 5), (2, 5), (4, 5), (5, 6), (7, 6), (8, 6), (9, 7), (10, 7), (11, 8), (12, 8)]
    print(unique_edges)      
    nodes = flatten_and_unique(unique_edges)
                       
    good_set = [10,11, 7]
    
    # initialize graph
    G, nstart = core_rank.init_graph(good_set, unique_edges)

    nx.draw(G, with_labels=True)
    plt.savefig("graph.png")
    
    # CORE BASED RANK 

    page_rank = core_rank.pagerank(G)
    page_rank_sorted = sorted(page_rank.items(), key = operator.itemgetter(1), reverse = True)
    print(' page rank modified -> core rank from nx')
    for k, v in page_rank_sorted:
        print(f'{k} --> {v}')
    print("-"*20)
    print('page rank from nx library')
    page_rank_nx = nx.pagerank(G)
    for k, v in page_rank_nx.items():
        print(f'{k} --> {v} ')
    print('-'*20)
    print('core rank custom implemented')
       
    corebased_rank = corerank.core_rank(unique_edges, good_set)


if __name__ == "__main__":
    main()