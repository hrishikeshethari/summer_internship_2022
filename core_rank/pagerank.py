import operator
import math, random, sys, csv 
import networkx as nx

class PageRank:
    def __init__(self, graph, directed, good_nodes):
        self.graph = graph
        self.V = len(self.graph)
        self.d = 0.85
        self.directed = directed
        self.ranks = dict()
        self.good_nodes = good_nodes
    
    def rank(self):
        for key, node in self.graph.nodes(data=True):
            #breakpoint()  
            if self.directed and key in self.good_nodes:
                self.ranks[key] = 1/float(self.V)
            else:
                self.ranks[key] = 0
             
        # for k, v in self.ranks.items():
        #     print(f'{k} --> {v}')
            
        for _ in range(100):
            for key, node in self.graph.nodes(data=True):
                rank_sum = 0
                curr_rank = node.get('rank')
                if self.directed:
                    neighbors = self.graph.edges(key)
                    for n in neighbors:
                        outlinks = len(self.graph.edges(n[1]))
                        if outlinks > 0:
                            rank_sum += (1 / float(outlinks)) * self.ranks[n[1]]
                else: 
                    neighbors = self.graph[key]
                    for n in neighbors:
                        if self.ranks[n] is not None:
                            outlinks = len([_ for _ in self.graph.neighbors(n)])
                            rank_sum += (1 / float(outlinks)) * self.ranks[n]
            
                # actual page rank compution
                self.ranks[key] = ((1 - float(self.d)) * (1/float(self.V))) + self.d*rank_sum

        return self.ranks
    
    def print_result(self):
        ranks = self.rank()
        sorted_rank = sorted(ranks.items())
        print('-'*20)
        for node, rank in sorted_rank:
            print(f'{node} --> {rank}')
        print('-'*20)

if __name__ == '__main__':
    p = PageRank(graph, isDirected)
    p.rank()

    sorted_r = sorted(p.ranks.iteritems(), key=operator.itemgetter(1), reverse=True)

    for tup in sorted_r:
        print('{0:30} :{1:10}'.format(str(tup[0]), tup[1]))

