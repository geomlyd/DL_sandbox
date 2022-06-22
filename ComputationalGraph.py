from multiprocessing.sharedctypes import Value
from GraphNode import GraphNode
import graphlib

class ComputationalGraph():

    def __init__(self):
        self.nameToNode = {}
        self.nodeToName = {}
        self.edgeTable = {}
        self.trainableNodes = []
        self.graphIterator = None
        self.topoSortUpToDate = False

    def getNode(self, nodeName : str):
        if(nodeName not in self.nameToNode):
            raise ValueError("No node named \"" + nodeName + "\" found in graph")
        return self.nameToNode[nodeName]

    def addNode(self, n : GraphNode, nodeName : str):
        if(nodeName in self.nameToNode.keys()):
            raise ValueError("A node named \"" + nodeName + "\" is already in the graph.")
        if(n in self.nodeToName.keys()):
            raise ValueError("Node\"" + nodeName + "\" is already in the graph under the same or different name")
        
        self.topoSortUpToDate = False
        self.nameToNode[nodeName] = n
        self.nodeToName[n] = nodeName
        if(n.isTrainable):
            self.trainableNodes.append(n)

    def runForwardPass(self):
        if(not self.topoSortUpToDate):
            for endNode in self.nodeToName.keys():
                endNodeName = self.nodeToName[endNode]
                for sourceNode in endNode.inEdges:
                    sourceNodeName = self.nodeToName[sourceNode]
                    if(sourceNodeName in self.edgeTable):
                        self.edgeTable[sourceNodeName].add(endNodeName)
                    else:
                        self.edgeTable[sourceNodeName] = set({endNodeName})
            try:
                self.graphIterator = tuple(graphlib.TopologicalSorter(self.edgeTable).static_order())
            except graphlib.CycleError:
                raise Exception("Computational graph has a cycle")  
            self.topoSortUpToDate = True         

        for nodeName in reversed(self.graphIterator):
            self.nameToNode[nodeName].forwardPass()

    def runBackwardPass(self):

        for nodeName in self.graphIterator:
            self.nameToNode[nodeName].backwardPass()