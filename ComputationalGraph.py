from abc import abstractmethod
from multiprocessing.sharedctypes import Value
from platform import node
from GraphNode import GraphNode
from typing import Dict
import graphlib
import Optimizer

class ComputationalGraph():

    def __init__(self, optimizer : Optimizer =None):
        self.nameToNode = {}
        self.nodeToName = {}
        self.edgeTable = {}
        self.trainableNodes = []
        self.trainOnlyNodes = set({})
        self.graphIterator = None
        self._optimizer = optimizer
        self.topoSortUpToDate = False

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, o):
        self._optimizer = o        

    def getNode(self, nodeName : str):
        if(nodeName not in self.nameToNode):
            raise ValueError("No node named \"" + nodeName + "\" found in graph")
        return self.nameToNode[nodeName]

    def addNode(self, n : GraphNode, nodeName : str, trainOnly : bool = False):
        if(nodeName in self.nameToNode.keys()):
            raise ValueError("A node named \"" + nodeName + "\" is already in the graph.")
        if(n in self.nodeToName.keys()):
            raise ValueError("Node\"" + nodeName + "\" is already in the graph under the same or different name")
        
        self.topoSortUpToDate = False
        self.nameToNode[nodeName] = n
        self.nodeToName[n] = nodeName
        if(n.isTrainable):
            self.trainableNodes.append(n)
        if(trainOnly):
            self.trainOnlyNodes.add(nodeName)


    def runForwardPass(self, runTraining=True):
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
                print("Computational graph has a cycle")
                exit(-1)
            self.topoSortUpToDate = True         

        for nodeName in reversed(self.graphIterator):
            if(runTraining or nodeName not in self.trainOnlyNodes):
                self.nameToNode[nodeName].forwardPass()

    def clearGradients(self):
        for nodeName in self.graphIterator:
            self.nameToNode[nodeName].clearGradients()

    def runBackwardPass(self):
        trainableParamGradientDict : Dict[GraphNode] = {} 
        for nodeName in self.graphIterator:
            n = self.nameToNode[nodeName]
            n.backwardPass()
            if(n.isTrainable and self._optimizer is not None):
                trainableParamGradientDict[n] = n.getParamGradient()

        if(self._optimizer is None):
            return
            
        trainableParamGradient = list(trainableParamGradientDict.values())
        paramStep = self._optimizer.computeStepWrapper(trainableParamGradient)
        i = 0
        for node in trainableParamGradientDict.keys():
            node.addToParamValues(paramStep[i])
            i += 1
        
        self.clearGradients()

