from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any

class EdgeBuffer():

    def __init__(self):
        self.buffer = None

    def writeToEdge(self, content):
        self.buffer = content

    def readFromEdge(self):
        return self.buffer


class GraphNode(ABC):

    def __init__(self):
        self.inEdges = {}
        self.outEdges = {}

    @abstractmethod
    def forwardPass(self):
        pass

    @abstractmethod
    def backwardPass(self):
        pass

    def registerInEdgeBuffer(self, edge : EdgeBuffer, name : str):
        if(name in self.inEdges.keys()):
            raise ValueError("Node already has an in-edge named \"" + name, "\"")
        self.inEdges[name] = edge
        
    def registerOutEdgeBuffer(self, edge : EdgeBuffer, name : str):
        if(name in self.outEdges.keys()):
            raise ValueError("Node already has an out-edge named \"" + name, "\"")        
        self.outEdges[name] = edge

    def getExistingEdgeBuffer(self, name : str):
        if(name not in self.outEdges.keys()):
            return None
        return self.outEdges[name]

