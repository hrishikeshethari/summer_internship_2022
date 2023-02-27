from typing import List, Tuple
import itertools
import matplotlib.pyplot as plt 
import networkx as nx

def init_graph(good_core_nodes: List, edges: List[Tuple]):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    nodes = set(itertools.chain(*edges))
    number_of_nodes = len(nodes)
    nstart_corebased = {idx : 0  for idx in range(number_of_nodes)}
    for idx, node in enumerate(nodes):
        if node in good_core_nodes:
            nstart_corebased[idx] = 1/number_of_nodes
    
    return G, nstart_corebased
        
def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    """ 
    taken from networkx source code, 
    only change is in nstart which initialization of random jump distribution i.e
    
    V+ = 1/n if good core node otherwise 0
    """
    if len(G) == 0:
        return {}

    D = G.to_directed()

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = sum(nstart.values())
        x = {k: v / s for k, v in nstart.items()}

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = sum(personalization.values())
        p = {k: v / s for k, v in personalization.items()}

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        s = sum(dangling.values())
        dangling_weights = {k: v / s for k, v in dangling.items()}
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        if isinstance(alpha, dict):
            alpha = list(alpha.values())[0]
        danglesum = alpha* sum(xlast[n] for n in dangling_nodes)
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for _, nbr, wt in W.edges(n, data=weight):
                x[nbr] += alpha * xlast[n] * wt
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        # check convergence, l1 norm
        err = sum(abs(x[n] - xlast[n]) for n in x)
        if err < N * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)

def main():
    import random 
    nodes = [_ for _ in range(10)]
        
    edges = [(random.choice(nodes), random.choice(nodes)) for _ in range(20)]
    unique_edges = []
    for edge in edges:
        if edge[0] != edge[1]:
            unique_edges.append(edge)        
    good_core_nodes = [1, 2, 3]
    G, nstart = init_graph(good_core_nodes, unique_edges)
    nx.draw(G, with_labels=True)
    plt.savefig("graph.png")
    corebased_rank = pagerank(G, nstart)
    core_sorted = sorted(corebased_rank.items())
    page_rank = pagerank(G)
    page_rank_sorted = sorted(page_rank.items())
    print('page rank')
    for k, v in page_rank_sorted:
        print(k, v)
    print("-"*20)
    print('core rank')       
    for k, v in core_sorted:
        print(k, v)

if __name__ == "__main__":
    main()