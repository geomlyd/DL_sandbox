from torch import Graph
from GraphNode import GraphNode, EdgeBuffer
import numpy as np

class InputNode(GraphNode):

    def __init__(self):
        super().__init__()
        self.content = None

    def setValue(self, v):
        self.content = v

    def forwardPass(self):
        for _, edge in self.outEdges.items():
            edge.writeToEdge(self.content)

    def backwardPass(self):
        pass

    def registerInEdgeBuffer(self, edge: EdgeBuffer, name: str):
        raise Exception("Input nodes cannot have in-edges")

class OutputNode(GraphNode):

    def __init__(self):
        super().__init__()
        self.content = None

    def forwardPass(self):
        self.content = {}
        for edgeName, edge in self.inEdges.items():
            self.content[edgeName] = edge.readFromEdge()
        
    def getValue(self):
        return self.content

    def backwardPass(self):
        pass

    def registerOutEdgeBuffer(self, edge: EdgeBuffer, name: str):
        raise Exception("Output nodes cannot have out-edges")

class ConstantNode(GraphNode):

    def __init__(self):
        super().__init__()
        self.content = None    

    def setValue(self, v):
        self.content = v

    def forwardPass(self):
        for _, edge in self.outEdges.items():
            edge.writeToEdge(self.content)

    def backwardPass(self):
        pass

    def registerInEdgeBuffer(self, edge: EdgeBuffer, name: str):
        raise Exception("Constant nodes cannot have in-edges")

class Add(GraphNode):

    def forwardPass(self):
        v = 0
        for _, edge in self.inEdges.items():
            v += edge.readFromEdge()
        for _, edge in self.outEdges.items():
            edge.writeToEdge(v)
        self.v = v

    def backwardPass(self):
        pass

class PointwiseMul(GraphNode):

    def forwardPass(self):
        v = 1
        for _, edge in self.inEdges.items():
            v *= edge.readFromEdge()
        for _, edge in self.outEdges.items():
            edge.writeToEdge(v)

    def backwardPass(self):
        pass


class PointwiseDivide(GraphNode):
 
    def forwardPass(self):
        if("numerator" not in self.inEdges or "denominator" not in self.inEdges):
            raise Exception("PointwiseDivide is lacking a \"numerator\" or a \"denominator\" edge")

        assert(len(self.inEdges) == 2)
        v = np.divide(self.inEdges["numerator"].readFromEdge(), self.inEdges["denominator"].readFromEdge())
     
        for _, edge in self.outEdges.items():
            edge.writeToEdge(v)

    def backwardPass(self):
        pass

    def registerInEdgeBuffer(self, edge: EdgeBuffer, name: str):
        if(len(self.inEdges) == 2):
            raise ValueError("PointwiseDivide node can only have 2 in-edges")
        super().registerInEdgeBuffer(edge, name)       
        

class Square(GraphNode):

    def forwardPass(self):
        assert(len(self.inEdges) == 1)
        for _, edge in self.inEdges.items():
            v = np.square(edge.readFromEdge())
        for _, edge in self.outEdges.items():
            edge.writeToEdge(v)

    def backwardPass(self):
        pass

    def registerInEdgeBuffer(self, edge: EdgeBuffer, name: str):
        if(len(self.inEdges) != 0):
            raise Exception("Log operation can only have one in-edge")
        super().registerInEdgeBuffer(edge, name)

class Log(GraphNode):

    def forwardPass(self):
        assert(len(self.inEdges) == 1)
        for _, edge in self.inEdges.items():
            v = np.log(edge.readFromEdge())        
        for _, edge in self.outEdges.items():
            edge.writeToEdge(v)

    def backwardPass(self):
        pass

    def registerInEdgeBuffer(self, edge: EdgeBuffer, name: str):
        if(len(self.inEdges) != 0):
            raise Exception("Square operation can only have one in-edge")
        super().registerInEdgeBuffer(edge, name)

class Sin(GraphNode):

    def forwardPass(self):
        assert(len(self.inEdges) == 1)
        for _, edge in self.inEdges.items():
            v = np.sin(edge.readFromEdge())        
        for _, edge in self.outEdges.items():
            edge.writeToEdge(v)

    def backwardPass(self):
        pass

    def registerInEdgeBuffer(self, edge: EdgeBuffer, name: str):
        if(len(self.inEdges) != 0):
            raise Exception("Sin operation can only have one in-edge")
        super().registerInEdgeBuffer(edge, name)
