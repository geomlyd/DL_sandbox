import ComputationalGraph
import CommonNodes
import numpy as np
import torch 
import ExampleDatasets
import torchvision.datasets as datasets

G = ComputationalGraph.ComputationalGraph()

v = np.array([[1, 2, 3], [4, 5, 6]])
i = CommonNodes.InputNode(v)
c = CommonNodes.InputNode(np.array([1, 2]))

s = CommonNodes.LogSoftmax(i)
l = CommonNodes.NegativeLogLikelihoodLoss(s, c)
o = CommonNodes.OutputNode(l)

G.addNode(i, "input")
G.addNode(c, "classes")
G.addNode(s, "logsoftmax")
G.addNode(l, "nll")
G.addNode(o, "out")

G.runForwardPass()
G.runBackwardPass()
print(o.value, i.totalGradient)

x = torch.tensor(v).float()
cc = torch.tensor(np.array([1, 2]))
x.requires_grad = True
y = torch.nn.LogSoftmax(dim=1)
z = torch.nn.NLLLoss()
w = z(y(x), cc)
w.backward()
print(w, x.grad)
#z = torch.mean(y, dim = )

d = ExampleDatasets.MNISTDataset("./test")
e = datasets.MNIST("./test2", download=True, train=True)