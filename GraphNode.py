from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any

from torch import Graph

# class EdgeBuffer():

#     def __init__(self):
#         self.buffer = None

#     def writeToEdge(self, content):
#         self.buffer = content

#     def readFromEdge(self):
#         return self.buffer


class GraphNode(ABC):

    def __init__(self, isTrainable=False, trackGradients=True):
        self.gradients = []
        self.totalGradient = 0
        self._value = None
        self._inEdges = []
        self._isTrainable = isTrainable
        self.trackGradients = trackGradients
        # self.inEdges = {}
        # self.outEdges = {}

    @abstractmethod
    def forwardPass(self):
        pass

    @abstractmethod
    def backwardPass(self):
        pass

    @property
    def isTrainable(self):
        return self._isTrainable

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v
    
    def addToValue(self, toBeAdded):
        self._value += toBeAdded

    def receiveGradient(self, grad):
        #self.gradients.append(grad)
        self.totalGradient += grad

    def getGradient(self):
        return self.gradients
    
    def setTrackGradients(self, trackGradients):
        self.trackGradients = trackGradients

    def registerInEdges(self, sourceNodes : list[GraphNode]):
        self._inEdges += sourceNodes

    @property
    def inEdges(self):
        return self._inEdges
    
    # def registerInEdgeBuffer(self, edge : EdgeBuffer, name : str):
    #     if(name in self.inEdges.keys()):
    #         raise ValueError("Node already has an in-edge named \"" + name, "\"")
    #     self.inEdges[name] = edge
        
    # def registerOutEdgeBuffer(self, edge : EdgeBuffer, name : str):
    #     if(name in self.outEdges.keys()):
    #         raise ValueError("Node already has an out-edge named \"" + name, "\"")        
    #     self.outEdges[name] = edge

    # def getExistingEdgeBuffer(self, name : str):
    #     if(name not in self.outEdges.keys()):
    #         return None
    #     return self.outEdges[name]

