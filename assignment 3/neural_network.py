import sys
import random
import math

LEARNING_RATE = 0.1
MOMENTUM_CONSTANT = 0.9
NUM_ITERATIONS = 20
NUM_INPUT_NODES = 2
NUM_HIDDEN_NODES = 2
NUM_OUTPUT_NODES = 1
ATTRIBUTES = {}
TARGET = []

class Node:
    def __init__(self, sigmoid):
        self.net = 0
        self.sigmoid = sigmoid
    def reset_node(self):
        self.net = 0
    def get_val(self):
        if self.sigmoid:
            return self.sigmoid_function(self.net)
        return self.net
    def add_val(self, val):
        self.net += val

    def sigmoid_function(self, val):
        return 1 / (1 + math.exp(-val))
    def getDerivative(self):
        if self.sigmoid:
            s = self.sigmoid_function(self.net)
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
                total[j] += node[i] * weights[i][j]
        return total
    
    def get_output(self, inputNode, outputNode = False):
        inputNode.append(1) # include a constant
        
        out = self.matrix_multiply(inputNode, self.inputWeights, self.numInputNodes + 1, self.numHiddenNodes)
        for i, val in enumerate(out):
            self.hiddenNodes[i].add_val(val)
        hidden = [self.hiddenNodes[i].get_val for i in range(self.numHiddenNodes)]
        hidden.append(1) # include a constant

        out = self.matrix_multiply(hidden, self.outputNodes, self.numHiddenNodes + 1, self.numOutputNodes)
        for i, val in enumerate(out):
            self.outputNodes[i].add_val(val)
        
        if outputNode:
            return self.outputNodes
        return [self.outputNodes[i].get_val for i in range(self.numOutputNodes)]

    def back_propogate(self, input, expectedOut):
        self.reset_nodes()
        out = self.get_output(input, True)

        deltaError = [(expectedOut[i] - out[i].get_val()) * out[i].getDerivative() for i in range(self.numOutputNodes)]

        # backpropogation to the hidden nodes
        deltahidden = [0 for _ in range(self.numHiddenNodes + 1)]
        for i in range(self.numHiddenNodes + 1):
            deltahidden[i] = self.hiddenNodes[i].getDerivative() * sum([deltaError[j] * self.hiddenWeights[i][j] for j in range(self.numOutputNodes)])

        for i in range(self.numHiddenNodes + 1):
            for j in range(self.numOutputNodes):
                if i == self.numHiddenNodes:
                    self.hiddenWeights[i][j] += self.learningRate * deltahidden[j] * self.hiddenNodes[i].get_val()
                else: 
                    self.hiddenWeights[i][j] += self.learningRate * deltahidden[j] # * 1  constant w0

        # repeat backprogogation to the input node
        deltaInput = [0 for _ in range(self.numInputNodes + 1)]
        for i in range(self.numInputNodes + 1):
            deltaInput[i] = self.inputWeights[i].getDerivative() * sum([deltahidden[j] * self.inputWeights[i][j] for j in range(self.numHiddenNodes + 1)])
        
        for i in range(self.numInputNodes + 1):
            for j in range(self.numHiddenNodes + 1):
                if i == self.numInputNodes:
                    self.inputWeights[i][j] += self.learningRate * deltaInput[j] * self.hiddenNodes[j].get_val()
                else:
                    self.inputWeights[i][j] += self.learningRate * deltaInput[j] # * 1  constant w0
    
    def output_nodes_and_weights(self):
        hiddenLength = 0
        for i in range(self.numInputNodes + 1):
            txt = str(["{:.1f}".format(i) for i in self.inputWeights[i]])
            if i < self.numHiddenNodes:
                txt += "  [" + "{:.1f}".format(self.hiddenNodes[i].get_val()) + "]  "
            else:
                txt += "         "
            if i < self.numHiddenNodes + 1:
                hiddenStr = str(["{:.1f}".format(i) for i in self.hiddenWeights[i]])
                if len(hiddenStr) > hiddenLength:
                    hiddenLength = len(hiddenStr)
                txt += hiddenStr
            else:
                txt += (" " * hiddenLength)
            
            if i < self.numOutputNodes:
                txt += "  [" + "{:.1f}".format(self.outputNodes[i].get_val()) + "]"
            print(txt)

def read_settings_file(file):
    global LEARNING_RATE, MOMENTUM_CONSTANT, NUM_ITERATIONS, NUM_HIDDEN_NODES
    with open(file, "r") as f:
        for line in f.readlines():
            if '=' not in line:
                continue
            name,val = line.split("=")
            val  = val.strip()
            if 'Hidden' in name:
                NUM_HIDDEN_NODES = int(val)
            elif 'Learning' in name:
                LEARNING_RATE = float(val)
            elif 'Momentum' in name:
                MOMENTUM_CONSTANT =float(val)
            elif 'Iterations' in name:
                NUM_ITERATIONS = int(val)


def read_attributes_file(file):
    global NUM_INPUT_NODES, NUM_OUTPUT_NODES, ATTRIBUTES, TARGET
    target = False
    first = True
    cnt = 0
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == '':
                target = True
                continue
            if not target:
                att = line.split()
                if ':' in att[0]:
                    cnt += 1
                    ATTRIBUTES[att[0]] = ['binary']
                else:
                    cnt += len(att) - 1
                    ATTRIBUTES[att[0]] = att[1:]
            else:
                if first and len(lines) == i + 1:
                    TARGET = [i.strip() for i in line.split()[1:]]
                else:
                    first = False
                    TARGET.append( [i.strip() for i in line.split()[1:]])
    NUM_OUTPUT_NODES = len(TARGET)
    NUM_INPUT_NODES = cnt

def read_data_file(file):
    global ATTRIBUTES
    dataList = []
    attList = list(ATTRIBUTES.keys())
    with open(file, "r") as f:
        for line in f.readlines():
            if '  ' in line:
                data = {}
                instance = []
                inputs, outputs = line.split('  ')
                print(inputs)
                for i, val in enumerate(inputs.split()):
                    if len (ATTRIBUTES[attList[i]]) == 1:
                        instance.append(val)
                    else:
                        attVals = [ 0 for _ in range(len(ATTRIBUTES[attList[i]]))]
                        for i, item in enumerate(ATTRIBUTES[attList[i]]):
                            if item in val:
                                attVals[i] = 1
                                break
                        instance.extend(attVals)
                data['attributes'] = instance
                data['target'] = outputs.split()
                dataList.append(data)
            else:
                data = {}
                instance = []
                inputs = line.split()
                for i, val in enumerate(inputs):
                    if i+1 == len(inputs):
                        break
                    elif len (ATTRIBUTES[attList[i]]) == 1:
                        instance.append(val)
                    else:
                        attVals = [ 0 for _ in range(len(ATTRIBUTES[attList[i]]))]
                        for i, item in enumerate(ATTRIBUTES[attList[i]]):
                            if item in val:
                                attVals[i] = 1
                                break
                        instance.extend(attVals)
                data['attributes'] = instance
                if ':' in inputs[-1]:
                    data['target'] = 'binary'
                else:
                    data['target'] = inputs[-1]
                dataList.append(data)
    return dataList

if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("No arguments provided.")
        exit(-1)
    
    settingsFile = sys.argv[1]
    read_settings_file(settingsFile)
    print('Number of Hiden Nodes: ', NUM_HIDDEN_NODES)
    print('Learning Rate: ', LEARNING_RATE)
    print('Momentum Constant: ', MOMENTUM_CONSTANT)
    print('Number of Iterations: ', NUM_ITERATIONS)

    attributeFile = sys.argv[2]
    read_attributes_file(attributeFile)
    print(ATTRIBUTES)
    print('Number of Input Nodes: ', NUM_INPUT_NODES)
    print('Number of Output Nodes: ', NUM_OUTPUT_NODES)
    print('Target: ', TARGET)

    dataFile = sys.argv[3]
    trainingSet = read_data_file(dataFile)
    for data in trainingSet:
        print(data)

    net = Network(NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, LEARNING_RATE, MOMENTUM_CONSTANT)
    net.output_nodes_and_weights()  

    
    