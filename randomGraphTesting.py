from RandomGraphGenerator import DAG, DAGToComputationalGraph
import CommonNodes

def inputNodeCreator(inputNodeList):
    assert(len(inputNodeList) == 0)
    return CommonNodes.InputNode()

def outputNodeCreator(inputNodeList):
    assert(len(inputNodeList) == 1)
    return CommonNodes.OutputNode(inputNodeList[0])

def sinNodeCreator(inputNodeList):
    assert(len(inputNodeList) == 1)
    return CommonNodes.Sin(inputNodeList[0])

def logNodeCreator(inputNodeList):
    assert(len(inputNodeList) == 1)
    return CommonNodes.Log(inputNodeList[0])

def subtractNodeCreator(inputNodeList):
    assert(len(inputNodeList) == 2)
    return CommonNodes.Subtract(inputNodeList[0], inputNodeList[1])

g = DAG(3, 5, 2, 0.5)
d = DAGToComputationalGraph("input", "output")

d.registerNode("input", inputNodeCreator, 0)
d.registerNode("output", outputNodeCreator, 1)
d.registerNode("sin", sinNodeCreator, 1)
d.registerNode("log", logNodeCreator, 1)
d.registerNode("sub", subtractNodeCreator, 2)

compGraph = d.convertDAGToCompGraph(g)
x = 1