from functools import total_ordering
from typing import List
from xmlrpc.client import boolean
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
        if(self.trackGradients):
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

class Subtract(GraphNode):

    def __init__(self, subtractFrom : GraphNode = None, valueToSubtract : GraphNode = None):
        super().__init__()
        self.subtractFrom = subtractFrom
        self.valueToSubtract = valueToSubtract 
        self.registerInEdges([subtractFrom, valueToSubtract])

    def setSubtractFrom(self, n : GraphNode):
        self.subtractFrom = n

    def setValueToSubtract(self, n : GraphNode):
        self.setValueToSubtract = n

    def forwardPass(self):
        self.value = self.subtractFrom.value - self.valueToSubtract.value

    def backwardPass(self):
        self.subtractFrom.receiveGradient(self.totalGradient)
        self.valueToSubtract.receiveGradient(-self.totalGradient)            

class Square(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
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
        self.producer = producer   
        self.registerInEdges([producer])

    def setProducer(self, p : GraphNode):
        self.producer = p

    def forwardPass(self):
        self.value = np.sin(self.producer.value)

    def backwardPass(self):
        self.producer.receiveGradient(self.totalGradient*np.cos(self.producer.value))
        
class ReLU(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.producer = producer
        self.registerInEdges([producer])

    def setProducer(self, p : GraphNode):
        self.producer = p   

    def forwardPass(self):
        v = self.producer.value
        self.value = np.copy(v)
        self.value[self.value < 0] = 0.01*self.value[self.value < 0]

    def backwardPass(self):
        g = np.zeros(self.totalGradient.shape) + 0.01
        g[self.value > 0] = 1
        self.producer.receiveGradient(g*self.totalGradient)


class AffineTransformation(GraphNode):

    def __init__(self, inputDimension : int, outputDimension : int, producer : GraphNode = None, 
        W_init : np.array = None , b_init : np.array = None):        
        super().__init__(isTrainable=True)
        self.producer = producer   
        if(W_init is not None):
            if(W_init.shape != (inputDimension, outputDimension)):
                raise Exception("AffineTransformation node: W initializer does not match declared dimensions")
            self.W = W_init.copy()
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
        v = self.producer.value
        if(len(v.shape) == 1):
            v = v[:, None]
        self.value = np.squeeze(np.matmul(v, self.W)+ self.b)

    def addToParamValues(self, paramStep):
        self.W = self.W + paramStep[0:self.W.shape[0]*self.W.shape[1]].reshape(self.W.shape)
        self.b = self.b + paramStep[self.W.shape[0]*self.W.shape[1]:]

    def backwardPass(self):
        if(len(self.totalGradient.shape) == 1):
            self.totalGradient = self.totalGradient[:, None]
        self.producer.receiveGradient(np.matmul(self.totalGradient, self.W.T))

        if(self.isTrainable):
            self.paramGradients = []
            self.paramGradients.append((np.matmul(self.producer.value.T, self.totalGradient).flatten()))
            self.paramGradients.append((np.sum(self.totalGradient, axis=0)))
            self.paramGradients = np.concatenate(self.paramGradients)

class ReduceSum(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.producer = producer
        if(producer is not None):
            self.registerInEdges([producer])

    def setProducer(self, p : GraphNode):
        self.producer = p
        if(len(self.inEdges) > 0):
            self.inEdges = []
        if(p is not None):
            self.registerInEdges([p])

    def forwardPass(self):
        self.value = np.sum(self.producer.value)

    def backwardPass(self):
        self.producer.receiveGradient(self.totalGradient*np.ones(self.producer.value.shape))

class LogSoftmax(GraphNode):

    def __init__(self, producer : GraphNode = None):
        super().__init__()
        self.producer = producer
        if(producer is not None):
            self.registerInEdges([producer])

    def setProducer(self, p : GraphNode):
        self.producer = p
        if(len(self.inEdges) > 0):
            self.inEdges = []
        if(p is not None):
            self.registerInEdges([p])

    def forwardPass(self):
        expValue = np.exp(self.producer.value)
        self.cache = expValue
        self.value = self.producer.value - np.log(np.sum(expValue, axis=1))[:, None]

    def backwardPass(self):
        dy_dx = self.cache/np.sum(self.cache, axis=1)[:, None]
        dy_dx = 1- dy_dx.shape[1]*dy_dx
        self.producer.receiveGradient(self.totalGradient*dy_dx)

class NegativeLogLikelihoodLoss(GraphNode):

    def __init__(self, logits : GraphNode = None, classes : GraphNode = None):
        super().__init__()
        self.logits = logits
        self.classes = classes
        self.registerInEdges([logits, classes])

    def forwardPass(self):
        booleanMask = False*np.ones(self.logits.value.shape)
        booleanMask = booleanMask.astype(bool)
        booleanMask[np.arange(booleanMask.shape[0]), self.classes.value.astype(int)] = True
        self.booleanMask = booleanMask
        self.value = np.sum(-self.logits.value[booleanMask])/self.classes.value.shape[0]

    def backwardPass(self):
        dy_dx = np.zeros(self.logits.value.shape)
        dy_dx[self.booleanMask] = 1/self.classes.value.shape[0]
        self.logits.receiveGradient(self.totalGradient*dy_dx)