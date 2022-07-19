from typing import List
from Transform import Transform


class Compose(Transform):

    def __init__(self, transformList : List[Transform]):
        self.transformList = transformList

    def __call__(self, x):
        y = x
        for t in self.transformList:
            y = t(y)
        return y

# class Normalize(Transform):

#     def __init__(self, )