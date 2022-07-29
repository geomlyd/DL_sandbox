from platform import node
from typing import Any
import numpy as np
import graphlib

from ComputationalGraph import ComputationalGraph
import CommonNodes


class DAG():

    def __init__(self, numLayers : int, maxNodesPerLayer : int, maxInDegree : int, 
        edgeProbability : float):

        self.edgeTable = {}
        self._inputVertices = set({})
        self._outputVertices = set({})
        self.inConnections = {}

        prevVertices = []
        for l in range(numLayers):
            numVertices = np.random.randint(maxNodesPerLayer + 1)
            layerVertices = ["l" + str(l) + "_n" + str(v) for v in range(numVertices)]
            if(l > 0):
                newEdges = np.random.random((len(prevVertices), numVertices))
                ind = newEdges < edgeProbability
                newEdges[ind] = 1
                newEdges[np.logical_not(ind)] = 0

                tmp = np.cumsum(newEdges, axis=0)
                newEdges[tmp > maxInDegree] = 0
                
                newEdgesByNames = [(prevVertices[i], layerVertices[j]) 
                    for j in range(numVertices) for i in range(len(prevVertices)) if newEdges[i, j]]
                print(newEdgesByNames)
                assert(len(newEdgesByNames) == np.sum(newEdges))
                
                
                for edge in newEdgesByNames:
                    if(edge[0] not in self.edgeTable):
                        self.edgeTable[edge[0]] = {edge[1]}
                    else:
                        self.edgeTable[edge[0]].add(edge[1])
                    
                    if(edge[1] not in self.inConnections):
                        self.inConnections[edge[1]] = {edge[0]}
                    else:
                        self.inConnections[edge[1]].add(edge[0])


            prevVertices += layerVertices

        self._graphIterator = tuple(graphlib.TopologicalSorter(self.edgeTable).static_order())

        for v in self.inConnections:
            if(len(self.inConnections[v]) == 0):
                self._inputVertices.add(v)
        for v in self.edgeTable:
            if(v not in self.edgeTable and len(self.inConnections[v]) > 0):
                self._outputVertices.add(v)

    @property
    def graphIterator(self):
        return self._graphIterator

    @property
    def inputVertices(self):
        return self._inputVertices

    @property
    def outputVertices(self):
        return self._outputVertices

    def getInDegree(self, nodeName):
        return len(self.inConnections[nodeName])

    def getInConnections(self, nodeName):
        return self.inConnections[nodeName]

class GraphNodeInformation():
    def __init__(self):
        self.nodesByInDegree = {}
        self.allNodes = set({})

    def addNode(self, nodeName : str, inDegree : int):

        if(nodeName in self.allNodes):
            raise ValueError("A node named \"" + nodeName + "\" is already registered")

        if inDegree not in self.nodesByInDegree:
            self.nodesByInDegree[inDegree] = set({nodeName})
        else:
            self.nodesByInDegree[inDegree].add(nodeName)
        self.allNodes.add(nodeName)

    def getNodesByInDegree(self, inDegree : int):
        return self.nodesByInDegree[inDegree]

class DAGToComputationalGraph():

    def __init__(self, inputNodeName, outputNodeName):
        self.nodeNamesToConstructors = {}
        self.inputNodeName = inputNodeName
        self.outputNodeName = outputNodeName
        self.nodeInformation = GraphNodeInformation()

    def registerNode(self, nodeName : str, constructorFunc : Any, inDegree : int):
        self.nodeInformation.addNode(nodeName, inDegree)

        self.nodeNamesToConstructors[nodeName] = constructorFunc

    def convertDAGToCompGraph(self, dag : DAG):
        graphIterator = dag.getGraphIterator()
        G = ComputationalGraph()
        graphOutputNodenames = []
        graphInputNodeNames = []

        for vertex in reversed(graphIterator):
            deg = dag.getInDegree(vertex)
            possibleFuncs = list(self.nodeInformation.getNodesByInDegree(deg))

            i = np.random.randint(len(possibleFuncs))

            if(possibleFuncs[i][0] == self.outputNodeName):
                graphOutputNodenames.append(vertex)
            elif(possibleFuncs[i][0] == self.inputNodeName):
                graphInputNodeNames.append(vertex)

            f = possibleFuncs[i][1]
            inConnections = dag.getInConnections(vertex)
            inNodes = [G.getNode(_) for _ in inConnections]
            G.addNode(vertex, f(inNodes))

        
        return G



DAG(5, 20, 2, 0.1)

                