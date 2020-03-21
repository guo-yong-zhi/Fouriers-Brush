import numpy as np


def DFStraverse(graph, node, unvisited):
    visited = set()
    
    def visitNode(node):
        visited.add(node)
        unvisited.remove(node)
        for child in graph(node):
            if child in unvisited:
                visitNode(child)
    visitNode(node)
    
    return visited, unvisited

from collections import deque

def BFStraverse(graph, node, unvisited):
    visited = set()
    queue = deque()
    
    queue.append(node)
    visited.add(node)
    unvisited.remove(node)
    
    while queue:
        n = queue.popleft()
        
        for child in graph(n):
            if child not in visited:
                queue.append(child)
                visited.add(child)
                unvisited.remove(child)
    
    return visited, unvisited

def BFS(graph, node, target):
    visited = {}
    queue = deque()
    
    queue.append(node)
    visited[node]=None
    
    found = False
    while queue and not found:
        n = queue.popleft()
        
        for child in graph(n):
            if child not in visited:
                queue.append(child)
                visited[child] = n
                if child == target:
                    found = True
    return found, visited

def backtrack(node, history):

    def track(trace):
        ac = history[trace[-1]]
        if ac is not None:
            trace.append(ac)
#            print("child, father", trace[-1], trace[-2])
        return ac
    
    trace = [node]
    while track(trace) is not None:
        pass
    
    return trace
          

def buildGraph(verts, edges):
    graph = {v:set() for v in verts}
    for a,b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph

class graph:
    def __init__(self, edges, vertices=None):
        self._edges = np.unique(np.array(edges),axis=0)
        if vertices is not None:
            self._vertices = vertices 
        else:
            self._vertices = np.unique(np.array(edges))
                
        self._vertices.sort()
        self.graph = None
        
        self.buildGraph()
        
        self.changed = False
        
        self._equivalence = []
        
    def copy(self):
        return graph(self.edges, self.vertices)
            
    def buildGraph(self):
        self.graph = buildGraph(self._vertices, self._edges)
    
    def neisOf(self, v):
        return self.graph[v]
    
    def __call__(self, v):
        return self.neisOf(v)

    def remove_edge(self, v1, v2):
        self.graph[v1].remove(v2)
        self.graph[v2].remove(v1)
        self.changed = True
    
    def add_edge(self, v1, v2):
        self.graph[v1].add(v2)
        self.graph[v2].add(v1)
        self.changed = True
    
    def refresh(self):
        es = []
        for v1,neis in self.graph.items():
            for v2 in neis:
                if v1 < v2:
                    es.append((v1, v2))
        self._edges = np.array(es)
        self.changed = False

    @property
    def edges(self):
        if self.changed:
            self.refresh()
        return self._edges
    
    @property
    def vertices(self):
        return self._vertices
    
    def partition(self):
        unvisited = set(self.vertices)
        while unvisited:
            visited, unvisited = BFStraverse(self, next(iter(unvisited)),
                                             unvisited)
            self._equivalence.append(visited)
    @property
    def equivalence(self):
        if self._equivalence:
            return self._equivalence
        else:
            self.buildGraph()
            self.partition()
            return self._equivalence
        
    def search(self, begin, target):
        if not self.graph:
            self.buildGraph()
        found, history = BFS(self, begin, target)
        trace = backtrack(target, history) if found else []
        return trace
    
class graph_adj:
    def __init__(self, edges=None, vertices=None, adjmat=None):
        if adjmat is not None :
            self.adjmat = adjmat
            self.shape = adjmat.shape
            self._vertices = np.arange(self.shape[0])
            self.refresh()
        else:
            self._vertices_o = vertices
            self._edges_o = edges
            
            if self._vertices_o is None and edges is not None:
                self._vertices_o = np.unique(np.array(edges))
                
            if self._vertices_o is not None:
                self._vertices_o.sort()
                
                self.shape = (len(self._vertices_o),) * 2
                
                self.maptoID()
                self.buildGraph()
        self.changed = False
        
    def copy(self):
        gc = graph_adj(adjmat=self.adjmat.copy())
        return gc 
        
    def maptoID(self):
        self.id2ver = dict(enumerate(self._vertices_o))
        self.ver2id = ver2id = {v:k for k,v in self.id2ver.items()}
        self._vertices = [ver2id[v] for v in self._vertices_o]
        self._vertices = np.array(self._vertices)
        self._edges = [(ver2id[v1], ver2id[v2]) for v1,v2 in self._edges_o]
        self._edges = np.array(self._edges)
    
    def buildGraph(self):
        self.adjmat = np.zeros(self.shape, dtype=np.int8)
        self.adjmat[(*self._edges.T,)] = 1
        self.adjmat[(*self._edges.T,)[::-1]] = 1
        
    def neisOf(self, v):
        neis = self.adjmat[v]
        return self.vertices[neis!=0]
    
    def __call__(self, v):
        return self.neisOf(v)
    
    def remove_edge(self, v1, v2):
        self.adjmat[v1,v2] = 0
        self.adjmat[v2,v1] = 0
        self.changed = True
    
    def add_edge(self, v1, v2):
        self.adjmat[v1,v2] = 1
        self.adjmat[v2,v1] = 1
        self.changed = True    
    
    def refresh(self):
        es = []
        for i1, r in enumerate(self.adjmat):
            for i2, c in enumerate(r):
                if c != 0 and i1<=i2:
                    es.append((i1,i2))
        self._edges = np.array(es)
        self.changed = False
                    
    @property
    def edges(self):
        if self.changed:
            self.refresh()
        return self._edges
    
    @property
    def vertices(self):
        return self._vertices
    
from scipy.sparse.csgraph import floyd_warshall

def isconnected_Floyd_Warshall(arr):
    "复杂度：O(n**3)，arr为邻接矩阵"
    dist = floyd_warshall(arr, directed=False, unweighted=True)
    return not np.isinf(dist).any()

def isconnected_BFS(g):
    "复杂度：O(n+m)，当g采用连接表存储"
    vs = set(g.vertices)
    v,uv = BFStraverse(g, next(iter(vs)), vs)
    return len(vs) == 0

import random 
def spanningTree_BrkCir_Floyd_Warshall(graph):
    "低效,O(n**4)"
    G = graph.copy()
    edges_in_circle = list(G.edges)
    while edges_in_circle:
#        print("edges num: ", len(edges_in_circle))
        ie = random.randint(0, len(edges_in_circle)-1) 
        v1, v2 = edges_in_circle.pop(ie)
        G.remove_edge(v1, v2)
        if not isconnected_Floyd_Warshall(G.adjmat):
            G.add_edge(v1, v2)
    return G

def spanningTree_BrkCir_BFS(graph):
    """
    破圈法生成最小生成树
    """
    def check_in_circle(v, G):
        "检测删除一些明显不在圈中的edge，未能降低复杂度，约节约一半时间"
        neis = G(v)
        if len(neis) == 1:
            nei = next(iter(neis))
#            print(v, "has one nei ", nei)
#            sucess = False
            try:
                edges_in_circle.remove((v, nei))
#                sucess = True
            except:pass
            try:
                edges_in_circle.remove((nei,v))
#                sucess = True
            except:pass          
#            assert(sucess)
            G.remove_edge(v, nei)
#            print(len(G.edges), "edges in circle(graph)")
#            print(len(edges_in_circle), "edges in circle(list)")
            check_in_circle(nei, G)
            
    G = graph.copy()
    edges_in_circle = [tuple(e) for e in G.edges]#list(G.edges)
    G2 = G.copy()
    while edges_in_circle:
#        print("edges num: ", len(edges_in_circle))
        ie = random.randint(0, len(edges_in_circle)-1) 
        v1, v2 = edges_in_circle.pop(ie)
        G.remove_edge(v1, v2)
        G2.remove_edge(v1, v2)
        if not isconnected_BFS(G):
            G.add_edge(v1, v2)
            G2.add_edge(v1,v2)
        else:    
            check_in_circle(v1, G2)
            check_in_circle(v2, G2)
        
    return G    

def spanningTree_path_len(graph, start, end, length):
    """
    破圈法生成最小生成树，指定start与end间距离
    """
    G = graph.copy()
    edges_in_circle = [tuple(e) for e in G.edges]#list(G.edges)
    expand_path = True
    while edges_in_circle:
#        print("edges num: ", len(edges_in_circle))        
        if expand_path:
            path = G.search(start, end)
#            print("current length:", len(path), "\texpected:", length)
            if len(path) >= length:
                #goal achieved
                expand_path = False
            else:
                path_i = list(range(len(path)-1))
                while path_i:
    #                print(way)
    #                print(G.edges)
                    i = random.randint(0, len(path_i)-1)
                    ip = path_i.pop(i)
                    v1, v2 = path[ip],path[ip+1]
                    G.remove_edge(v1, v2)
    
                    try:
                        edges_in_circle.remove((v1, v2))
                    except:pass
                    try:
                        edges_in_circle.remove((v2, v1))
                    except:pass
                
                    if not isconnected_BFS(G):
                        G.add_edge(v1, v2)
                    else: 
                        break
                else:
                    #expanding faild
                    expand_path = False
                    print("the path expanding faild!")
                    print("path length:", len(path))
                    print("path:", path)
                    print("start,end:", start,end)
               
        else:
            ie = random.randint(0, len(edges_in_circle)-1) 
            v1, v2 = edges_in_circle.pop(ie)
            G.remove_edge(v1, v2)
            if not isconnected_BFS(G):
                G.add_edge(v1, v2)
                
    print("The final length of the path is %s,"%len(G.search(start, end)),
          "expected:", length)
  
    return G  


if __name__ == "__main__":
    es = [("a","b"),("c","d"),("d","f"),("b","e"),("e","d"),("f","a"),("c","f")]
    g = graph_adj(es)
    t = spanningTree_BrkCir_Floyd_Warshall(g)
    
    g2 = graph(es)
    t2 = spanningTree_BrkCir_BFS(g2)
    
    spanningTree_path_len(g2, "a", "f", 3)
#    print(isconnected_BFS(g2))