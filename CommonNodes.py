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
        self.registerInEdges([producer])

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
        self.registerInEdges(producers)

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
        self.registerInEdges(producers)

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
        self.registerInEdges([numerator, denominator])

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
        self.registerInEdges([producer])

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
        self.registerInEdges([producer])

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
        self.registerInEdges([producer])

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = np.sin(self.producer.value)

    def backwardPass(self):
        self.producer.receiveGradient(self.totalGradient*np.cos(self.producer.value))
        pass

class AffineTransformation(GraphNode):

    def __init__(self, inputDimension : int, outputDimension : int, producer : GraphNode = None, 
        W_init : np.array = None , b_init : np.array = None):        
        super().__init__(isTrainable=True)
        self.value = None
        self.producer = producer   
        if(W_init is not None):
            if(W_init.shape != (inputDimension, outputDimension)):
                raise Exception("AffineTransformation node: W initializer does not match declared dimensions")
            self.W = W_init
        else:
            self.W = np.zeros((inputDimension, outputDimension))
        if(b_init is not None):
            if(W_init.shape != (inputDimension, outputDimension)):
                raise Exception("AffineTransformation node: b initializer does not match declared dimensions")
            self.b = b_init
        else:
            self.b = np.zeros(outputDimension)
        self.registerInEdges([producer])

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = np.matmul(self.producer.value, self.W)+ self.b

    def backwardPass(self):
        self.producer.receiveGradient(np.matmul(self.totalGradient, self.W.T))

        if(self.isTrainable):
            self.paramGradients.append(np.matmul(self.producer.value.T, self.totalGradient))
            self.paramGradients.append(np.sum(self.totalGradient, axis=0))