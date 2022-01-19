from abc import ABC, abstractmethod


class GraphNode(ABC):

    @abstractmethod
    def forwardPass(self):
        pass

    @abstractmethod
    def backwardPass(self):
        pass

class ComputationalGraph(ABC):

    @abstractmethod
    def 