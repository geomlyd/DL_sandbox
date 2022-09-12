from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class GraphNode(ABC):

    def __init__(self, isTrainable=False, trackGradients=True):
        self.gradients = []
        self.totalGradient = 0
        self._value = None
        self._inEdges = []
        self._isTrainable = isTrainable
        self._trackGradients = trackGradients
        self._paramGradients = []

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

    @property
    def inEdges(self):
        return self._inEdges

    @inEdges.setter
    def inEdges(self, inEdges):
        self._inEdges = inEdges

    @property
    def trackGradients(self):
        return self._trackGradients

    @property
    def paramGradients(self):
        return self._paramGradients

    @trackGradients.setter
    def trackGradients(self, trackGradients):
        self._trackGradients = trackGradients    

    def clearGradients(self):
        self.paramGradients = []
        self.totalGradient = 0
        self.gradients = []

    def receiveGradient(self, grad):
        self.totalGradient += grad

    def registerInEdges(self, sourceNodes : list[GraphNode]):
        self._inEdges += sourceNodes

    def addToParamValues(self, paramStep):
        pass
