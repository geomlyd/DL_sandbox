from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any

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
        self.value = None
        self.isTrainable=isTrainable
        self.trackGradients=trackGradients
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
        return self.isTrainable

    def getValue(self):
        return self.value
    
    def addToValue(self, toBeAdded):
        self.value += toBeAdded

    def receiveGradient(self, grad):
        self.gradients.append(grad)

    def getGradient(self):
        return self.gradients
    
    def setTrackGradients(self, trackGradients):
        self.trackGradients = trackGradients

    
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

