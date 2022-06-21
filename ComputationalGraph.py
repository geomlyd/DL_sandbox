from GraphNode import GraphNode
import graphlib

class ComputationalGraph():

    def __init__(self):
        self.nodeTable = {}
        self.edgeTable = {}
        self.trainableNodes = []
        self.graphIterator = None

    def getNode(self, nodeName : str):
        if(nodeName not in self.nodeTable):
            raise ValueError("No node named \"" + nodeName + "\" found in graph")
        return self.nodeTable[nodeName]

    def addNode(self, n : GraphNode, nodeName : str):
        if(nodeName in self.nodeTable.keys()):
            raise ValueError("A node named \"" + nodeName + "\" is already in the graph.")

        self.nodeTable[nodeName] = n
        if(n.isTrainable):
            self.trainableNodes.append(n)

    def addEdge(self, sourceNode : str, endNode : str):
        if(sourceNode not in self.nodeTable.keys()):
            raise ValueError("Graph has no node named \"" + sourceNode)
        if(endNode not in self.nodeTable.keys()):
            raise ValueError("Graph has no node named \"" + endNode)

        if(sourceNode in self.edgeTable):
            self.edgeTable[sourceNode].add(endNode)
        else:
            self.edgeTable[sourceNode] = set({endNode})
        
        try:
            self.graphIterator = tuple(graphlib.TopologicalSorter(self.edgeTable).static_order())
        except graphlib.CycleError:
            print("Computational graph has a cycle")
            exit()

    def runForwardPass(self):

        for node in reversed(self.graphIterator):
            self.nodeTable[node].forwardPass()

    def runBackwardPass(self):

        for node in self.graphIterator:
            self.nodeTable[node].backwardPass()