import random
import math

INPUT_NODES = 1

class Node:
    def __init__(self, sigmoid = False):
        self.net = 0
        self.sigmoid = sigmoid
    def reset_node(self):
        self.net = 0
    def get_val(self)
        if self.sigmoid:
            return sigmoid_function(self.net)
        return self.net
    def add_val(self, val):
        self.net += val
    def sigmoid_function(self, val):
        return 1 / (1 + math.exp(-val))
    def getDerivative(self):
        if self.sigmoid:
            s = sigmoid_function(self.net)
            return s(1-s)
        return 1

class Network:
    def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes, hiddenSigmoid = False, outputSigmoid = False):
        self.numInputNodes = numInputNodes
        self.numHiddenNodes = numHiddenNodes
        self.numOutputNodes = numOutputNodes
        self.inputWeights = [[random.random() for _ in range(numHiddenNodes)] for _ in range(numInputNodes)]
        self.hiddenNodes = [Node(hiddenSigmoid) for _ in range(numHiddenNodes)]
        self.hiddenWeights = [[random.random() for _ in range(numOutputNodes)] for _ in range(numHiddenNodes)]
        self.outputNodes = [Node(outputSigmoid) for _ in range(numOutputNodes)]
    
    def matrix_multiply(self, node, weights, nodelength, outputlength):
        total = [0 for _ in range(outputlength)]
        for i in range(nodelength):
            for j in range(outputlength):
                total[j] += node[j] weight[i][j]
        return total
    
    def get_output(self, inputNode):
        out = self.matrix_multiply(inputNode, self.inputWeights, self.numInputNodes, self.numHiddenNodes)
        for i, val in enumerate(out):
            self.hiddenNodes[i].add_val(val)
        hidden = [self.hiddenNodes[i].get_val for i in range(self.numHiddenNodes)]

        out = self.matrix_multiply(hidden, self.outputNodes, self.numHiddenNodes, self.numOutputNodes)
        for i, val in enumerate(out):
            self.outputNodes[i].add_val(val)
        return [self.outputNodes[i].get_val for i in range(self.numOutputNodes)]
        