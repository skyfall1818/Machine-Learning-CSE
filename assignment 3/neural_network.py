import sys
import random
import math

#global variables
LEARNING_RATE = 0.1
MOMENTUM_CONSTANT = 0.9
NUM_ITERATIONS = 20
NUM_INPUT_NODES = 2
NUM_HIDDEN_NODES = 2
NUM_OUTPUT_NODES = 1
ATTRIBUTES = {}
TARGET = []
VALIDATION_PERCENTAGE = 0.2
MAX_NOISE_LEVEL = 20
NOISE_INCREMENT = 2

############################################
# class Network
# Holds the structure of a single node of the ANN
###############################################
class Node:
    # sigmoid is saved to use
    def __init__(self, sigmoid):
        self.net = 0
        self.sigmoid = sigmoid
    
    # sets the node to 0
    def reset_node(self):
        self.net = 0
    
    # returns the net value
    # includes sigmoid function is flag is true
    def get_val(self):
        if self.sigmoid:
            return self.sigmoid_function(self.net)
        return self.net
    
    # adds a value to the node
    def add_val(self, val):
        self.net += val

    # returns sigmoid function value
    def sigmoid_function(self, val):
        return 1 / (1 + math.exp(-val))
    
    # returns the derivative
    # if sigmoid ios true, returns teh derivative of teh sigmoid, else 1 
    def getDerivative(self):
        if self.sigmoid:
            s = self.sigmoid_function(self.net)
            return s * (1-s) # ds(x)/dx = s(x)[1-s(x)]
        return 1

############################################
# class Network
# Holds the entire ANN
# contains the weights, and nodes
# contains algorithms for training and learning
###############################################
class Network:

    # inputs the number of nodes need for each layer
    # hidden Signmoid and output sigmoid are optional, add the sigmoid function
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
        self.bestHiddenWeights = [[0 for _ in range(numOutputNodes)] for _ in range(numHiddenNodes + 1)] # adding +1 to include a constant w0
        self.bestInputWeights = [[0 for _ in range(numHiddenNodes)] for _ in range(numInputNodes + 1)] # adding +1 to include a constant w0

    # resets the nodes. sets all to 0 value
    def reset_nodes(self):
        for h in range(self.numHiddenNodes):
            self.hiddenNodes[h].reset_node()
        for o in range(self.numOutputNodes):
            self.outputNodes[o].reset_node()
    
    # multiplies two matrices
    # input node * the wights matrix
    def matrix_multiply(self, nodes, weights, nodelength, outputlength):
        total = [0 for _ in range(outputlength)]
        for i in range(nodelength):
            for j in range(outputlength):
                total[j] += nodes[i] * weights[i][j]
        return total
    
    # gets the predicted output from the network
    # outputNode is optional. its controls if we want to return the nodes or its values
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
            return self.outputNodes # returns the lise of nodes themselves. used for back progagation. saves computation time
        if self.outputNodes[0].sigmoid: # threshold output
            return [1 if self.outputNodes[i].get_val() >=0.5 else 0 for i in range(self.numOutputNodes)]
        best = 0
        bestIndex = -1
        for i,node in enumerate(self.outputNodes): # best node output
            if bestIndex == -1 or node.get_val() > best:
                bestIndex = i
                best = node.get_val()
        return [0 if i != bestIndex else 1 for i in range(self.numOutputNodes)]

    # Runs the main Backpropogation algorithm
    def back_propogate(self, input, expectedOut):
        if input is None or expectedOut is None:
            raise ValueError("Input and expected output cannot be None")
        
        if len(expectedOut) != self.numOutputNodes:
            raise ValueError("Expected output length must match number of output nodes")
        
        out = self.get_output(input, True)
        
        deltaError = [(expectedOut[i] - out[i].get_val()) * out[i].getDerivative() for i in range(self.numOutputNodes)]# output layer delta
        
        # Backpropagation to the hidden nodes
        deltahidden = [0 for _ in range(self.numHiddenNodes + 1)]# hidden layer delta
        for i in range(self.numHiddenNodes):
            deltahidden[i] = self.hiddenNodes[i].getDerivative() * sum(deltaError[j] * self.hiddenWeights[i][j] for j in range(self.numOutputNodes))
        deltahidden[self.numHiddenNodes] = sum(deltaError[j] * self.hiddenWeights[self.numHiddenNodes][j] for j in range(self.numOutputNodes))
        
        # updates the weights for the hidden to output layer
        for i in range(self.numHiddenNodes + 1):
            for j in range(self.numOutputNodes):
                if i < self.numHiddenNodes:
                    self.prevHiddenWeights[i][j] = self.prevHiddenWeights[i][j] * self.momentumCnst + self.learningRate * deltaError[j] * self.hiddenNodes[i].get_val() # momentum + learning rate * deltaError * hiddenNode
                    self.hiddenWeights[i][j] += self.prevHiddenWeights[i][j]
                else:
                    self.prevHiddenWeights[i][j] = self.prevHiddenWeights[i][j] * self.momentumCnst + self.learningRate * deltaError[j] # * 1 (constant w0)
                    self.hiddenWeights[i][j] += self.prevHiddenWeights[i][j]
        
        # updates the weights for the input to hidden layer
        for i in range(self.numInputNodes + 1):
            for j in range(self.numHiddenNodes):
                if i < self.numInputNodes:
                    self.prevInputWeights[i][j] = self.prevInputWeights[i][j] * self.momentumCnst + self.learningRate * deltahidden[j] * input[i] # momentum + learning rate * deltahidden * inputnode
                    self.inputWeights[i][j] += self.prevInputWeights[i][j]
                else:
                    self.prevInputWeights[i][j] = self.prevInputWeights[i][j] * self.momentumCnst + self.learningRate * deltahidden[j] # * 1 (constant w0)
                    self.inputWeights[i][j] += self.prevInputWeights[i][j]
    
    # Prints the weights 
    # I_H means input to hidden layer
    # H_O means hidden to output layer
    def output_nodes_and_weights(self):
        print("Weights:")
        hiddenLength = 0
        txt = " "
        for i in range(self.numHiddenNodes):
            txt += str(" I_H{:4s}".format(str(i)))
        txt += "  "
        for i in range(self.numOutputNodes):
            txt += str(" H_O{:4s}".format(str(i)))
        print(txt)
        for i in range(self.numInputNodes + 1):
            txt = str(["{:4.1f}".format(i) for i in self.inputWeights[i]])
            
            txt += "  "

            if i < self.numHiddenNodes + 1:
                hiddenStr = str(["{:4.1f}".format(i) for i in self.hiddenWeights[i]])
                if len(hiddenStr) > hiddenLength:
                    hiddenLength = len(hiddenStr)
                txt += hiddenStr
            else:
                txt += (" " * hiddenLength)
            print(txt)
    
    # Returns the accuracy of the network based on the given test set
    def get_accuracy(self, testSet):
        correct = 0
        for data in testSet:
            out = self.get_output(data['attributes'])
            if out == data['target']:
                correct += 1
        return correct / len(testSet)
    
    # Saves the best weights and accuracy to the validation set
    def update_best_weights(self, accuracy):
        if accuracy >= self.bestAccuracy:
            self.bestAccuracy = accuracy
            self.bestInputWeights = [[i for i in row] for row in self.inputWeights]
            self.bestHiddenWeights = [[i for i in row] for row in self.hiddenWeights]
    
    # Sets the network to the best saved weights
    def set_network_to_best_weights(self):
        self.inputWeights = [[ i for i in row] for row in self.bestInputWeights]
        self.hiddenWeights = [[ i for i in row] for row in self.bestHiddenWeights]

# reads the settings file
# looks for key words in each line
# order of the key words is not important
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

# reads the attributes file save it the the global variable ATTRIBUTES and TARGETS
def read_attributes_file(file):
    global NUM_INPUT_NODES, NUM_OUTPUT_NODES, ATTRIBUTES, TARGET
    target = False
    first = True
    cnt = 0
    with open(file, "r") as f: # reads the file
        lines = f.readlines()
        for i, line in enumerate(lines): # reads each line
            if line.strip() == '': # the first empty line means the next line is the target line
                target = True
                continue

            # This formats the inputs so it is easier to put into the network
            if not target: # test for target line
                att = line.split()
                if ':' in att[0]:
                    cnt += 1
                    ATTRIBUTES[att[0]] = ['binary'] # 1 or 0 output
                else:
                    cnt += len(att) - 1
                    ATTRIBUTES[att[0]] = att[1:]
            else: # saves the target
                if first and len(lines) == i + 1:
                    TARGET = [i.strip() for i in line.split()[1:]]
                else:
                    first = False
                    TARGET.append( [i.strip() for i in line.split()[1:]])
    NUM_OUTPUT_NODES = len(TARGET) # gets more info of the inputs structure for teh network
    NUM_INPUT_NODES = cnt

# reads the traninig set and test set files
# targets and attributes are staored based on how it will be implemented in the network
def read_data_file(file):
    global ATTRIBUTES
    dataList = []
    attList = list(ATTRIBUTES.keys())
    with open(file, "r") as f:
        for line in f.readlines():
            if '  ' in line: # catch to find the Identity test set
                data = {} # for each single instance we save thje attributes and the targets
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
            else: # this is for iris and tennis test sets
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
                data['target'] = [ 1 if val in inputs[-1] else 0 for val in TARGET] # theres should only be one target output since this is descrete variables
                dataList.append(data)
    return dataList

# splits the training set into a training set and a validation set
def get_Validation_Set(trainingSet, validationPercentage):
    validationSet = []
    trainingLength = len(trainingSet)
    for _ in range(int(trainingLength * validationPercentage)):
        validationSet.append(trainingSet.pop())
        trainingSet.pop(index)
    return validationSet

# slowly adds noise to the training set. validation set is included
def add_noise(trainingSet, validationSet, noiseNum):
    global ATTRIBUTES
    trainingLength = len(trainingSet) + len(validationSet)
    for _ in range(noiseNum):
        index = random.randint(0, trainingLength - 1) # random selecting a training example
        attIndex = random.randint(0, len(ATTRIBUTES) - 1) # random selecting an attribute to add noise
        att = list(ATTRIBUTES.keys())[attIndex]
        if index < len(trainingSet): # training set noise 
            if ATTRIBUTES[att] == ['binary']: # since binary, swap bits
                if trainingSet[index]['attributes'][attIndex] == 0:
                    trainingSet[index]['attributes'][attIndex] = 1
                else:
                    trainingSet[index]['attributes'][attIndex] = 0
            elif ATTRIBUTES[att] == ['continuous']: # uiform didistribution random number change between -10 and 10
                trainingSet[index]['attributes'][attIndex] = random.uniform(-10, 10)
            else:
                for i in ATTRIBUTES[att]:
                    if trainingSet[index]['attributes'][attIndex] == i:
                        while trainingSet[index]['attributes'][attIndex] == i: # repeats until it gets a different value
                            trainingSet[index]['attributes'][attIndex] = ATTRIBUTES[att][random.randint(0, len(ATTRIBUTES[att]) - 1)]
                        break
        else: #validation set noise
            index = index - len(trainingSet)
            if ATTRIBUTES[att] == ['binary'] :# since binary, swap bits
                if validationSet[index]['attributes'][attIndex] == 0:
                    validationSet[index]['attributes'][attIndex] = 1
                else:
                    validationSet[index]['attributes'][attIndex] = 0
            elif ATTRIBUTES[att] == ['continuous']: # uiform didistribution random number change between -10 and 10
                validationSet[index]['attributes'][attIndex] = random.uniform(-10, 10)
            else:
                for i in ATTRIBUTES[att]: # descrete attributes
                    if validationSet[index]['attributes'][attIndex] == i:
                        while validationSet[index]['attributes'][attIndex] == i: # repeats until it gets a different value
                            validationSet[index]['attributes'][attIndex] = ATTRIBUTES[att][random.randint(0, len(ATTRIBUTES[att]) - 1)]
                        break

# main code
if __name__ == "__main__":
    random.seed('MLCSE')
    if len(sys.argv) <= 1:
        print("No arguments provided.")
        exit(-1)
    
    if len(sys.argv) < 5:
        print("Invalid number of arguments.")
        exit(-1)
    
    settingsFile = sys.argv[1] #argument 1
    read_settings_file(settingsFile)

    attributeFile = sys.argv[2] #argument 2
    read_attributes_file(attributeFile)

    trainFile = sys.argv[3] #argument 3
    trainingSet = read_data_file(trainFile)
    uncorruptSet = read_data_file(trainFile)
    
    testFile = sys.argv[4] #argument 4
    testingSet = read_data_file(testFile)

    # optional arguments
    noisy = False
    noTest = False
    valid = False
    verbose = False
    index = 5
    while index < len(sys.argv):
        if sys.argv[index] == '-n':
            noisy = True
        elif sys.argv[index] == '-noTest':
            noTest = True
        elif sys.argv[index] == '-v':
            verbose = True
        elif sys.argv[index] == '-valid':
            valid = True
        index += 1

    if valid: # validation set
        validationSet = get_Validation_Set(trainingSet, VALIDATION_PERCENTAGE)
    else:
        validationSet = []

    if not noisy: # no noise
        net = Network(NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, LEARNING_RATE, MOMENTUM_CONSTANT, True, True)
        print('Training Network...')
        for n in range(NUM_ITERATIONS):
            acc = 0
            for i in trainingSet: #training the network
                net.back_propogate(i['attributes'], i['target'])
            if noTest:
                acc = net.get_accuracy(testingSet)
                net.update_best_weights(acc)
            if valid:
                acc = net.get_accuracy(validationSet)
                net.update_best_weights(acc)
            if acc > 0.99: # stopping because we already found the best/ most accurate network and sets of weights
                break
        if valid or noTest:
            net.set_network_to_best_weights()
        print('Finished')
        if noTest:
            print('accuracy: ' + str(net.get_accuracy(uncorruptSet)))
        else:
            print('accuracy training: ' + str(net.get_accuracy(uncorruptSet)))
            print('accuracy test: ' + str(net.get_accuracy(testingSet)))
        if verbose:
            net.output_nodes_and_weights()
    else: # adding noise
        noiseLevel = 0.0
        totalNoiseCnt = 0
        total = len(trainingSet) + len(validationSet)
        while noiseLevel <= MAX_NOISE_LEVEL: # incrementing noise by 0.02
            net = Network(NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, LEARNING_RATE, MOMENTUM_CONSTANT, True, True)
            print('\n----Noise Level:' + str(noiseLevel) + '----')
            for _ in range(NUM_ITERATIONS):
                acc = 0
                for i in trainingSet: #training the network
                    net.back_propogate(i['attributes'], i['target'])
                if noTest:
                    acc = net.get_accuracy(testingSet)
                    net.update_best_weights(acc)
                if valid:
                    acc = net.get_accuracy(validationSet)
                    net.update_best_weights(acc)
                if acc > 0.99: # stopping because we already found the best/ most accurate network and sets of weights
                    break
            if valid or noTest:
                net.set_network_to_best_weights()
            if noTest:
                print('accuracy: ' + str(net.get_accuracy(uncorruptSet)))
            else:
                print('accuracy training: ' + str(net.get_accuracy(uncorruptSet)))
                print('accuracy test: ' + str(net.get_accuracy(testingSet)))
            if verbose:
                net.output_nodes_and_weights()
            noiseLevel += NOISE_INCREMENT
            noiseNum = int(total * noiseLevel)
            # adds the small amount of noise tot he test set
            add_noise(trainingSet,validationSet, noiseNum - totalNoiseCnt)
            totalNoiseCnt = noiseNum