from typing import List
import numpy as np
from Transform import Transform


class Compose(Transform):

    def __init__(self, transformList : List[Transform]):
        self.transformList = transformList

    def __call__(self, x):
        y = x
        for t in self.transformList:
            y = t(y)
        return y

class NormalizeImage(Transform):

    def __init__(self, channelMeans : np.array, channelStds : np.array):
        self.channelMeans = np.array(channelMeans)
        self.channelStds = np.array(channelStds)

        if(len(self.channelMeans.shape) == 0):
            self.channelMeans = self.channelMeans[None]
        if(len(self.channelStds.shape) == 0):
            self.channelStds = self.channelStds[None]
        

    def __call__(self, x):
        return (x - self.channelMeans[:, None, None])/self.channelStds[:, None, None]