from functools import total_ordering
from typing import List
from torch import Graph
from GraphNode import GraphNode
import numpy as np

class InputNode(GraphNode):

    def __init__(self, value=None):
        super().__init__()

    def setValue(self, v):
        self.value = v

    def forwardPass(self):
        pass

    def backwardPass(self):
        pass


class OutputNode(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.producer = producer

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = self.producer.value

    def backwardPass(self):
        self.producer.receiveGradient(1)

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
            v += p.value()
        self.value = v

    def backwardPass(self):
        for p in self.producers:
            p.receiveGradient(self.totalGradient)
        

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
            v  *= p.value
        self.value = v

    def backwardPass(self):
        for p in self.producers:
            p.receiveGradient(self.totalGradient*np.divide(self.value, p.value))

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
        self.value = np.divide(self.numerator.value, self.denominator.value)

    def backwardPass(self):
        self.numerator.receiveGradient(np.divide(self.totalGradient, self.denominator.value))
        self.denominator.receiveGradient(np.divide(self.totalGradient*(-self.numerator.value), 
            self.denominator.value*self.denominator.value))          

class Square(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.value = None
        self.producer = producer   

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = np.square(self.producer.value)
        # if(self.trackGradients):
        #     self.cachedArg = self.producer.value()

    def backwardPass(self):
        #gradShape = self.gradients.shape
        #argShape = self.cachedArg.shape
        self.producer.receiveGradient(self.totalGradient*2*self.producer.value)

class Log(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.value = None
        self.producer = producer   

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = np.log(self.producer.value)

    def backwardPass(self):
        self.producer.receiveGradient(np.divide(self.totalGradient, self.producer.value))

class Sin(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.value = None
        self.producer = producer   

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = np.sin(self.producer.value)

    def backwardPass(self):
        self.producer.receiveGradient(self.totalGradient*np.cos(self.producer.value))
        pass
