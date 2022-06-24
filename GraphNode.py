from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List

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
        self.paramGradients = []
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

    @property
    def inEdges(self):
        return self._inEdges

    @value.setter
    def value(self, v):
        self._value = v

    def clearGradients(self):
        self.paramGradients = []
        self.totalGradient = 0
        self.gradients = []

    def receiveGradient(self, grad):
        #self.gradients.append(grad)
        self.totalGradient += grad

    def getParamGradient(self):
        return self.paramGradients
    
    def setTrackGradients(self, trackGradients):
        self.trackGradients = trackGradients

    def registerInEdges(self, sourceNodes : list[GraphNode]):
        self._inEdges += sourceNodes

    def addToParamValues(self, paramStep):
        pass
