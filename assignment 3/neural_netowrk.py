import random
import math

INPUT_NODES = 1

class Node:
    def __init__(self, sigmoid):
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
            return s(1-s) # ds(x)/dx = s(x)[1-s(x)]
        return 1

class Network:
    def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes, learningRate, momentumConstant, hiddenSigmoid = False, outputSigmoid = False):
        self.learningRate = learningRate
        self.momentumCnst = momentumConstant
        self.numInputNodes = numInputNodes
        self.numHiddenNodes = numHiddenNodes
        self.numOutputNodes = numOutputNodes
        self.inputWeights = [[random.random() for _ in range(numHiddenNodes)] for _ in range(numInputNodes + 1)] # adding +1 to include a constant w0
        self.hiddenNodes = [Node(hiddenSigmoid) for _ in range(numHiddenNodes)]
        self.hiddenWeights = [[random.random() for _ in range(numOutputNodes)] for _ in range(numHiddenNodes + 1)] # adding +1 to include a constant w0
        self.outputNodes = [Node(outputSigmoid) for _ in range(numOutputNodes)]

    def reset_nodes(self):
        for h in range(self.numHiddenNodes):
            self.hiddenNodes[i].reset_node()
        for o in range(self.numOutputNodes):
            self.outputNodes[o].reset_node()
    
    def matrix_multiply(self, node, weights, nodelength, outputlength):
        total = [0 for _ in range(outputlength)]
        for i in range(nodelength):
            for j in range(outputlength):
                total[j] += node[i] * weight[i][j]
        return total
    
    def get_output(self, inputNode):
        inputNode.append(1) # include a constant
        
        out = self.matrix_multiply(inputNode, self.inputWeights, self.numInputNodes + 1, self.numHiddenNodes)
        for i, val in enumerate(out):
            self.hiddenNodes[i].add_val(val)
        hidden = [self.hiddenNodes[i].get_val for i in range(self.numHiddenNodes)]
        hidden.append(1) # include a constant

        out = self.matrix_multiply(hidden, self.outputNodes, self.numHiddenNodes + 1, self.numOutputNodes)
        for i, val in enumerate(out):
            self.outputNodes[i].add_val(val)
        return [self.outputNodes[i].get_val for i in range(self.numOutputNodes)]

    def back_propogate(self, input, expectedOut):
        self.reset_nodes()
        out = self.get_output(input)
        
        
        
