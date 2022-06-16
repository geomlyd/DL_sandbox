from typing import List
from torch import Graph
from GraphNode import GraphNode
import numpy as np

class InputNode(GraphNode):

    def __init__(self, value=None):
        super().__init__()
        self.value = None

    def setValue(self, v):
        self.value = v

    def forwardPass(self):
        pass

    def backwardPass(self):
        pass


class OutputNode(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.value = None
        self.producer = producer

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = self.producer.getValue()

    def backwardPass(self):
        pass

class ConstantNode(GraphNode):

    def __init__(self, value=None):
        super().__init__()
        self.value = value    

    def forwardPass(self):
        pass

    def backwardPass(self):
        pass

class Add(GraphNode):

    def __init__(self, producers : List[GraphNode] = None):
        super().__init__()
        self.value = None
        self.producers = producers    

    def addProducer(self, p : GraphNode):
        self.producers.append(p)

    def forwardPass(self):
        v = 0
        for p in self.producers:
            v += p.getValue()
        self.value = v

    def backwardPass(self):
        pass

class PointwiseMul(GraphNode):

    def __init__(self, producers : List[GraphNode] = None):
        super().__init__()
        self.value = None
        self.producers = producers    

    def addProducer(self, p : GraphNode):
        self.producers.append(p)

    def forwardPass(self):
        v = 1
        for p in self.producers:
            v  = np.multiply(v, p.getValue())
        self.value = v

    def backwardPass(self):
        pass

class PointwiseDivide(GraphNode):

    def __init__(self, numerator : GraphNode = None, denominator : GraphNode = None):
        super().__init__()
        self.value = None
        self.numerator = numerator
        self.denominator = denominator 

    def setNumerator(self, n : GraphNode):
        self.numerator = n

    def setDenominator(self, d : GraphNode):
        self.denominator = d

    def forwardPass(self):
        self.value = np.divide(self.numerator.getValue(), self.denominator.getValue())

    def backwardPass(self):
        pass          

class Square(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.value = None
        self.producer = producer   

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = np.square(self.producer.getValue())

    def backwardPass(self):
        pass

class Log(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.value = None
        self.producer = producer   

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = np.log(self.producer.getValue())

    def backwardPass(self):
        pass

class Sin(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.value = None
        self.producer = producer   

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = np.sin(self.producer.getValue())

    def backwardPass(self):
        pass
