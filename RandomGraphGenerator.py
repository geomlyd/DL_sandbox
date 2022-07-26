from asyncio import proactor_events
import numpy as np
import graphlib


class DAG():

    def __init__(self, numLayers : int, maxNodesPerLayer : int, edgeProbability : float):
        self.edgeTable = {}

        prevVertices = []
        for l in range(numLayers):
            numVertices = np.random.randint(maxNodesPerLayer + 1)
            layerVertices = ["l" + str(l) + "_n" + str(v) for v in range(numVertices)]
            if(l > 0):
                newEdges = np.random.random((len(prevVertices), numVertices))
                ind = newEdges < edgeProbability
                newEdges[ind] = 1
                newEdges[np.logical_not(ind)] = 0
                
                newEdgesByNames = [(prevVertices[i], layerVertices[j]) 
                    for j in range(numVertices) for i in range(len(prevVertices)) if newEdges[i, j]]
                print(newEdgesByNames)
                assert(len(newEdgesByNames) == np.sum(newEdges))
                
                
                for edge in newEdgesByNames:
                    if(edge[0] not in self.edgeTable):
                        self.edgeTable[edge[0]] = {edge[1]}
                    else:
                        self.edgeTable[edge[0]].add(edge[1])

            prevVertices += layerVertices

        self.graphIterator = tuple(graphlib.TopologicalSorter(self.edgeTable).static_order())


DAG(5, 20, 0.1)

                