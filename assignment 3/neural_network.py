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
MAX_NOISE_LEVEL = 20
NOISE_INCREMENT = 2

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
            return s * (1-s) # ds(x)/dx = s(x)[1-s(x)]
        return 1

class Network:
    def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes, learningRate, momentumConstant, hiddenSigmoid = False, outputSigmoid = False):
        self.learningRate = learningRate
        self.momentumCnst = momentumConstant
        self.numInputNodes = numInputNodes
        self.numHiddenNodes = numHiddenNodes
        self.numOutputNodes = numOutputNodes

        self.inputWeights = [[random.uniform(-1, 1) for _ in range(numHiddenNodes)] for _ in range(numInputNodes + 1)] # adding +1 to include a constant w0
        self.hiddenNodes = [Node(hiddenSigmoid) for _ in range(numHiddenNodes)]
        self.hiddenWeights = [[random.uniform(-1, 1) for _ in range(numOutputNodes)] for _ in range(numHiddenNodes + 1)] # adding +1 to include a constant w0
        self.outputNodes = [Node(outputSigmoid) for _ in range(numOutputNodes)]
        self.prevInputWeights = [[0 for _ in range(numHiddenNodes)] for _ in range(numInputNodes + 1)] # adding +1 to include a constant w0
        self.prevHiddenWeights = [[0 for _ in range(numOutputNodes)] for _ in range(numHiddenNodes + 1)] # adding +1 to include a constant w0

        self.bestAccuracy = 0
        self.besthiddenWeights = [[0 for _ in range(numOutputNodes)] for _ in range(numHiddenNodes + 1)] # adding +1 to include a constant w0
        self.bestinputWeights = [[0 for _ in range(numHiddenNodes)] for _ in range(numInputNodes + 1)] # adding +1 to include a constant w0

    def reset_nodes(self):
        for h in range(self.numHiddenNodes):
            self.hiddenNodes[h].reset_node()
        for o in range(self.numOutputNodes):
            self.outputNodes[o].reset_node()
    
    def matrix_multiply(self, nodes, weights, nodelength, outputlength):
        total = [0 for _ in range(outputlength)]
        for i in range(nodelength):
            for j in range(outputlength):
                total[j] += nodes[i] * weights[i][j]
        return total
    
    def get_output(self, inputNode, outputNode = False):
        self.reset_nodes()
        inputNode.append(1) # include a constant
        
        out = self.matrix_multiply(inputNode, self.inputWeights, self.numInputNodes + 1, self.numHiddenNodes)
        inputNode.pop()
        for i, val in enumerate(out):
            self.hiddenNodes[i].add_val(val)
        hidden = [self.hiddenNodes[i].get_val() for i in range(self.numHiddenNodes)]
        hidden.append(1) # include a constant

        out = self.matrix_multiply(hidden, self.hiddenWeights, self.numHiddenNodes + 1, self.numOutputNodes)
        for i, val in enumerate(out):
            self.outputNodes[i].add_val(val)
        
        if outputNode:
            return self.outputNodes
        if self.outputNodes[0].sigmoid:
            return [1 if self.outputNodes[i].get_val() >=0.5 else 0 for i in range(self.numOutputNodes)]
        best = 0
        bestIndex = -1
        for i,node in enumerate(self.outputNodes):
            if bestIndex == -1 or node.get_val() > best:
                bestIndex = i
                best = node.get_val()
        return [0 if i != bestIndex else 1 for i in range(self.numOutputNodes)]

    def back_propogate(self, input, expectedOut):
        if input is None or expectedOut is None:
            raise ValueError("Input and expected output cannot be None")
        
        if len(expectedOut) != self.numOutputNodes:
            raise ValueError("Expected output length must match number of output nodes")
        
        out = self.get_output(input, True)
        
        deltaError = [(expectedOut[i] - out[i].get_val()) * out[i].getDerivative() for i in range(self.numOutputNodes)]
        
        # Backpropagation to the hidden nodes
        deltahidden = [0 for _ in range(self.numHiddenNodes + 1)]
        for i in range(self.numHiddenNodes):
            deltahidden[i] = self.hiddenNodes[i].getDerivative() * sum(deltaError[j] * self.hiddenWeights[i][j] for j in range(self.numOutputNodes))
        deltahidden[self.numHiddenNodes] = sum(deltaError[j] * self.hiddenWeights[self.numHiddenNodes][j] for j in range(self.numOutputNodes))
        
        for i in range(self.numHiddenNodes + 1):
            for j in range(self.numOutputNodes):
                if i < self.numHiddenNodes:
                    self.prevHiddenWeights[i][j] = self.prevHiddenWeights[i][j] * self.momentumCnst + self.learningRate * deltaError[j] * self.hiddenNodes[i].get_val() # momentum + learning rate * deltaError * hiddenNode
                    self.hiddenWeights[i][j] += self.prevHiddenWeights[i][j]
                else:
                    self.prevHiddenWeights[i][j] = self.prevHiddenWeights[i][j] * self.momentumCnst + self.learningRate * deltaError[j] # * 1 (constant w0)
                    self.hiddenWeights[i][j] += self.prevHiddenWeights[i][j]
        
        for i in range(self.numInputNodes + 1):
            for j in range(self.numHiddenNodes):
                if i < self.numInputNodes:
                    self.prevInputWeights[i][j] = self.prevInputWeights[i][j] * self.momentumCnst + self.learningRate * deltahidden[j] * input[i] # momentum + learning rate * deltahidden * inputnode
                    self.inputWeights[i][j] += self.prevInputWeights[i][j]
                else:
                    self.prevInputWeights[i][j] = self.prevInputWeights[i][j] * self.momentumCnst + self.learningRate * deltahidden[j] # * 1 (constant w0)
                    self.inputWeights[i][j] += self.prevInputWeights[i][j]
    
    def output_nodes_and_weights(self):
        hiddenLength = 0
        for i in range(self.numInputNodes + 1):
            txt = str(["{:.1f}".format(i) for i in self.inputWeights[i]])
            if i < self.numHiddenNodes:
                txt += "  [" + "{:2.1f}".format(self.hiddenNodes[i].get_val()) + "]  "
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
    
    def get_accuracy(self, testSet):
        correct = 0
        for data in testSet:
            out = self.get_output(data['attributes'])
            if out == data['target']:
                correct += 1
        return correct / len(testSet)
    
    def update_best_weights(self, accuracy):
        if accuracy > self.bestAccuracy:
            self.bestAccuracy = accuracy
            self.bestInputWeights = [[i for i in row] for row in self.inputWeights]
            self.bestHiddenWeights = [[i for i in row] for row in self.hiddenWeights]
    
    def set_network_to_best_weights(self):
        self.inputWeights = self.bestInputWeights
        self.hiddenWeights = self.bestHiddenWeights

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
                for i, val in enumerate(inputs.split()):
                    if len (ATTRIBUTES[attList[i]]) == 1:
                        instance.append(int(val))
                    else:
                        attVals = [ 0 for _ in range(len(ATTRIBUTES[attList[i]]))]
                        for i, item in enumerate(ATTRIBUTES[attList[i]]):
                            if item in val:
                                attVals[i] = 1
                                break
                        instance.extend(attVals)
                data['attributes'] = instance
                data['target'] = [int(i) for i in outputs.split()]
                dataList.append(data)
            else:
                data = {}
                instance = []
                inputs = line.split()
                for i, val in enumerate(inputs):
                    if i+1 == len(inputs):
                        break
                    elif len (ATTRIBUTES[attList[i]]) == 1:
                        instance.append(float(val))
                    else:
                        attVals = [ 0 for _ in range(len(ATTRIBUTES[attList[i]]))]
                        for i, item in enumerate(ATTRIBUTES[attList[i]]):
                            if item in val:
                                attVals[i] = 1
                                break
                        instance.extend(attVals)
                data['attributes'] = instance
                data['target'] = [ 1 if val in inputs[-1] else 0 for val in TARGET]
                dataList.append(data)
    return dataList

def add_noise(trainingSet, noisePercent):
    global ATTRIBUTES
    trainingLength = len(trainingSet)
    for _ in range(int(trainingLength * noisePercent)):
        index = random.randint(0, trainingLength - 1)
        attIndex = random.randint(0, len(ATTRIBUTES) - 1)
        att = list(ATTRIBUTES.keys())[attIndex]
        if ATTRIBUTES[att] == ['binary']:
            if trainingSet[index]['attributes'][attIndex] == 0:
                trainingSet[index]['attributes'][attIndex] = 1
            else:
                trainingSet[index]['attributes'][attIndex] = 0
        elif ATTRIBUTES[att] == ['continuous']:
            trainingSet[index]['attributes'][attIndex] += random.uniform(-1, 1)
        else:
            for i in ATTRIBUTES[att]:
                if trainingSet[index]['attributes'][attIndex] == i:
                    while trainingSet[index]['attributes'][attIndex] == i:
                        trainingSet[index]['attributes'][attIndex] = ATTRIBUTES[att][random.randint(0, len(ATTRIBUTES[att]) - 1)]
                    break

if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("No arguments provided.")
        exit(-1)
    
    if len(sys.argv) < 5:
        print("Invalid number of arguments.")
        exit(-1)
    
    settingsFile = sys.argv[1]
    read_settings_file(settingsFile)

    attributeFile = sys.argv[2]
    read_attributes_file(attributeFile)

    trainFile = sys.argv[3]
    trainingSet = read_data_file(trainFile)
    
    testFile = sys.argv[4]
    testingSet = read_data_file(testFile)

    noisy = False
    index = 5
    while index < len(sys.argv):
        if sys.argv[index] == '-n':
            noisy = True
        index += 1

    if not noisy:
        net = Network(NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, LEARNING_RATE, MOMENTUM_CONSTANT, True, True)
        print('Training Network...')
        cnt = 0
        for _ in range(NUM_ITERATIONS):
            acc = 0
            for i in trainingSet:
                net.back_propogate(i['attributes'], i['target'])
            acc = net.get_accuracy(testingSet)
            net.update_best_weights(acc)
            cnt += 1
            #if cnt % 10 == 0:
            #    print('Accuracy: ' + str(acc))
        net.set_network_to_best_weights()
        print('Finished')
        print('best weights')
        print('accuracy: ' + str(net.get_accuracy(testingSet)))
        net.output_nodes_and_weights()
    else:
        noiseLevel = 0.0
        while noiseLevel <= MAX_NOISE_LEVEL:
            net = Network(NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, LEARNING_RATE, MOMENTUM_CONSTANT, True, True)
            print('Noise Level: ' + str(noiseLevel))
            print('Training Network...')
            cnt = 0
            for _ in range(NUM_ITERATIONS):
                acc = 0
                for i in trainingSet:
                    net.back_propogate(i['attributes'], i['target'])
                acc = net.get_accuracy(testingSet)
                net.update_best_weights(acc)
                cnt += 1
            net.set_network_to_best_weights()
            print('Finished')
            print('best weights')
            print('accuracy: ' + str(net.get_accuracy(testingSet)))
            net.output_nodes_and_weights()
            noiseLevel += NOISE_INCREMENT
            add_noise(trainingSet, NOISE_INCREMENT/100)