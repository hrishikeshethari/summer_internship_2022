import networkx as nx
import numpy as np 
import math 
import matplotlib.pyplot as plt 
import core_rank
from pagerank import PageRank

def flatten_and_unique(l):
    return set([item for sublist in l for item in sublist])

def main():
    import random 
    nodes = [_ for _ in range(10)]
        
    edges = [(random.choice(nodes), random.choice(nodes)) for _ in range(20)]
    unique_edges = []
    for edge in edges:
        if edge[0] != edge[1]:
            unique_edges.append(edge)
            
    nodes = flatten_and_unique(unique_edges)
                       
    good_set = [1, 2, 3]
    
    # initialize graph
    G, nstart = core_rank.init_graph(good_set, unique_edges)
    # PAGERANK FOR COMPARISON  
    p = PageRank(G, True, good_set)
    print('Comparative pagerank different implementation')
    print(p.print_result())
    nx.draw(G, with_labels=True)
    plt.savefig("graph.png")
    # CORE BASED RANK 
    corebased_rank = core_rank.pagerank(G, nstart)
    core_sorted = sorted(corebased_rank.items())
    page_rank = core_rank.pagerank(G)
    page_rank_sorted = sorted(page_rank.items())
    print('page rank')
    for k, v in page_rank_sorted:
        print(f'{k} --> {v}')
    print("-"*20)
    print('core rank')       
    for k, v in core_sorted:
        print(f'{k} --> {v}')


if __name__ == "__main__":
    main()