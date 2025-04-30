import sys
import random
import math
import time

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
ALPHA = 2
BETA = 2
DEGREE = 3
DEGREE_INCREMENT = 0
ALPHA_INCREMENT = 0
BETA_INCREMENT = 0
START_TIME = 0
END_TIME = 0
SEED = ''

############################################
# class Network
# Holds the structure of a single node of the ANN
###############################################
class Node:
    # We will use the jacobi recurrence formula
    def __init__(self, tan_jacobi, sigmoid = False, alpha = -1, beta = -1, n = 0):
        self.net = 0
        self.tan_jacoi = tan_jacobi
        self.sigmoid = sigmoid
        self.alpha = alpha
        self.beta = beta
        self.Jn_abc = [] # holds the jacobi a, b, c values for each iteration/step for the recurrence formula
        self.Jn_abc_derivative = [] # holds the jacobi a, b, c for the derivative
        self.degree = n
        if tan_jacobi:
            self.set_jacobi(n, alpha, beta)
    
    # sets the node to 0
    def reset_node(self):
        self.net = 0
    
    # returns the net value
    # includes tanh and jacobi if the flag is true
    def get_val(self):
        if self.tan_jacoi:
            tanh = self.tanh_function(self.net)
            return self.jacobi_function(tanh)
        if self.sigmoid:
            return self.sigmoid_function(self.net)
        return self.net
    
    # returns the derivative 
    def getDerivative(self):
        if self.tan_jacoi:
            return self.tanh_derivative(self.net) * self.jacobi_derivative(self.net)
        if self.sigmoid:
            return self.sigmoid_derivative(self.net)
        return 1
    
    # adds a value to the node
    def add_val(self, val):
        self.net += val

    def sigmoid_function(self, val):
        if val < -100:
            return 0
        return 1 / (1 + math.exp(-val))
    
    def sigmoid_derivative(self, val):
        sig = self.sigmoid_function(val)
        return sig * (1 - sig)

    # returns the tanh function
    def tanh_function(self, val):
        return math.tanh(val)
    
    # returns the derivative of tanh. d(tanh(x))/dx = sech^2(x) = 1 - tanh^2(x)
    def tanh_derivative(self, val):
        tanh = self.tanh_function(val)
        return 1 - tanh * tanh

    # this sets the a,b, and c values for the recurrence formula
    # this is done during the building of the network to reduce computation time
    def set_jacobi(self, degree, alpha, beta):
        for n in range(degree):
            if n == 0: # 0th degree is a constant. dont need to caculate a, b, and c
                self.Jn_abc.append([0, 0, 0])
                continue
            elif n == 1: # 1st degree polynomial set for a, b, and c
                A0 = (alpha + beta + 2) / 2
                B0 = (alpha - beta)/ 2
                C0 = 0
                self.Jn_abc.append([A0, B0, C0])
                continue
            # calculates the three term relations of the recurrence formula
            An = (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2) / (2 * (n + 1) * (n + alpha + beta + 1) )
            Bn = (alpha * alpha - beta * beta) * (2 * n + alpha + beta + 1) / (2 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta) )
            Cn = (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2) / ( (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta) )
            self.Jn_abc.append([An, Bn, Cn])
        
        # need to calculate J{alpha + 1, beta + 1}(x) to calculate the derivative
        alpha += 1
        beta += 1
        for n in range(degree):
            if n == 0: # 0th degree is a constant. dont need to caculate a, b, and c
                self.Jn_abc_derivative.append([0, 0, 0])
                continue
            elif n == 1: # 1st degree polynomial set for a, b, and c
                A0 = (alpha + beta + 2) / 2
                B0 = (alpha - beta)/ 2
                C0 = 0
                self.Jn_abc_derivative.append([A0, B0, C0])
                continue
            # calculates the three term relations for the derivative
            An = (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2) / (2 * (n + 1) * (n + alpha + beta + 1) )
            Bn = (alpha * alpha - beta * beta) * (2 * n + alpha + beta + 1) / (2 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta) )
            Cn = (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2) / ( (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta) )
            self.Jn_abc_derivative.append([An, Bn, Cn])
    
    # returns the jacobi polynomial function using the recurrence formula
    # and extra argument for derivative allows us to switch to which function (set of a,b,c values) we need to use
    def jacobi_function(self, val, derivative = False):
        if self.degree < 2:
            if self.degree == 0:
                return 1 # 0th degree
            return 0.5 * (self.alpha + self.beta + 2) * val + 0.5 * (self.alpha - self.beta) # first degree
        elif not derivative: # calculate the jacobi polynomial
            jDegrees = [ 0 for _ in range(self.degree + 1)]
            jDegrees[0] = 1 # 0th degree
            jDegrees[1] = self.Jn_abc[1][0] * val + self.Jn_abc[1][1] # 1st degree
            for i in range(2, self.degree + 1):
                jDegrees[i] = (self.Jn_abc[i - 1][0] * val + self.Jn_abc[i - 1][1]) * jDegrees[i - 1] - self.Jn_abc[i - 1][2] * jDegrees[i - 2]
            return jDegrees[self.degree]
        else:  # calculate the jacobi polynomial recursion for the derivative
            jDegrees = [ 0 for _ in range(self.degree)]
            jDegrees[0] = 1 # 0th degree
            if self.degree > 1:
                jDegrees[1] = self.Jn_abc_derivative[1][0] * val + self.Jn_abc_derivative[1][1] # 1st degree
                for i in range(2, self.degree):
                    jDegrees[i] = (self.Jn_abc_derivative[i - 1][0] * val + self.Jn_abc_derivative[i - 1][1]) * jDegrees[i - 1] - self.Jn_abc_derivative[i - 1][2] * jDegrees[i - 2]
            return jDegrees[self.degree - 1]
    
    # return the derivative of the jacobi polynomial function 
    def jacobi_derivative(self, val):
        if self.degree == 0:
            print('returning a 0')
            return 0
        else:
            return 0.5 * (self.degree + self.alpha + self.beta + 1) * self.jacobi_function(val, True)


############################################
# class Network
# Holds the entire ANN
# contains the weights, and nodes
# contains algorithms for training and learning
###############################################
class Network:

    # inputs the number of nodes need for each layer
    def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes, learningRate, momentumConstant, IncreasingDegree, randomizeAnB, alpha = 0, beta = 0, degree = 0):
        self.learningRate = learningRate
        self.momentumCnst = momentumConstant
        self.numInputNodes = numInputNodes
        self.numHiddenNodes = numHiddenNodes
        self.numOutputNodes = numOutputNodes

        self.inputWeights = [[random.uniform(-1/(numInputNodes), 1/(numInputNodes)) for _ in range(numHiddenNodes)] for _ in range(numInputNodes + 1)] # adding +1 to include a constant w0

        #setting each hidden node to a set jacobi polynomial
        if randomizeAnB:
            self.hiddenNodes = [Node(True, False, random.randint(0, numHiddenNodes + 1), random.randint(0, numHiddenNodes + 1),random.randint(1, numHiddenNodes + 1)) for _ in range(numHiddenNodes)]
        else:
            self.hiddenNodes = [Node(True, False, alpha + IncreasingDegree[0] * k, beta + IncreasingDegree[1] * k, degree + IncreasingDegree[2] * k) for k in range(numHiddenNodes)]
        self.hiddenWeights = [[random.uniform(-0.1, 0.1) for _ in range(numOutputNodes)] for _ in range(numHiddenNodes + 1)] # adding +1 to include a constant w0
        self.outputNodes = [Node(False, True) for _ in range(numOutputNodes)]
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
            return self.outputNodes # returns the list of nodes themselves. used for back progagation. saves computation time
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
            txt += str("  I_H{:4s}".format(str(i)))
        txt += "  "
        for i in range(self.numOutputNodes):
            txt += str("  H_O{:4s}".format(str(i)))
        print(txt)
        for i in range(self.numInputNodes + 1):
            txt = str(["{:5.2f}".format(i) for i in self.inputWeights[i]])
            
            txt += "  "

            if i < self.numHiddenNodes + 1:
                hiddenStr = str(["{:5.2f}".format(i) for i in self.hiddenWeights[i]])
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
    global LEARNING_RATE, MOMENTUM_CONSTANT, NUM_ITERATIONS, NUM_HIDDEN_NODES, ALPHA, BETA, DEGREE, DEGREE_INCREMENT, ALPHA_INCREMENT, BETA_INCREMENT, SEED
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
            elif 'Iteration' in name:
                NUM_ITERATIONS = int(val)
            elif 'Alpha' in name:
                ALPHA = int(val)
            elif 'Beta' in name:
                BETA = int(val)
            elif 'Degree' in name:
                DEGREE = int(val)
            elif 'D Increment' in name:
                DEGREE_INCREMENT = int(val)
            elif 'A Increment' in name:
                ALPHA_INCREMENT = int(val)
            elif 'B Increment' in name:
                BETA_INCREMENT = int(val)
            elif 'Seed' in name:
                SEED = val

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

def start_time():
    global START_TIME
    START_TIME = time.time()

def stop_time():
    global END_TIME
    END_TIME = time.time()
    print_time()

def print_time():
    global START_TIME, END_TIME
    print("Took %s seconds\n" % (END_TIME - START_TIME))


# main code
if __name__ == "__main__":
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
    
    testFile = sys.argv[4] #argument 4
    testingSet = read_data_file(testFile)

    # optional arguments
    valid = False
    verbose = False
    recordTime = False
    threshold = False
    randomabn = False
    outputIterations = False
    index = 5
    while index < len(sys.argv):
        if sys.argv[index] == '-v':
            verbose = True
        elif sys.argv[index] == '-valid':
            valid = True
        elif sys.argv[index] == '-t':
            recordTime = True
        elif sys.argv[index] == '-th' or sys.argv[index] == '-threshold':
            threshold = True
        elif sys.argv[index] == '-r':
            randomabn = True
        elif sys.argv[index] == '-o':
            outputIterations = True
        else:
            print("Invalid argument: " + sys.argv[index])
            exit(-1)
        index += 1

    if SEED != '':
        print("Setting seed to: " + str(SEED))
        random.seed(SEED)

    if valid: # validation set
        validationSet = get_Validation_Set(trainingSet, VALIDATION_PERCENTAGE)
    else:
        validationSet = []
    
    increments = [ALPHA_INCREMENT, BETA_INCREMENT, DEGREE_INCREMENT]
    # setting up the network
    if recordTime:
        start_time()
        print('Building Network')
    net = Network(NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, LEARNING_RATE, MOMENTUM_CONSTANT, increments, randomabn, ALPHA, BETA, DEGREE)
    if recordTime:
        print('Done Building Network')
        stop_time()
        start_time()
    print('Training Network...')
    if not threshold:
        print("Number of iterations: " + str(NUM_ITERATIONS))
        for n in range(NUM_ITERATIONS):
            acc = 0
            for i in trainingSet: #training the network
                net.back_propogate(i['attributes'], i['target'])
            if valid:
                acc = net.get_accuracy(validationSet)
                net.update_best_weights(acc)
            if acc > 0.99 and not outputIterations: # stopping because we already found the best/ most accurate network and sets of weights
                break
            if outputIterations and (n+1) % 10 == 0:
                print(str(n + 1) + ' | accuracy: ' + str(acc) + '; time: ' + str(time.time() - START_TIME))
        if valid:
            net.set_network_to_best_weights()
    else:
        acc = 0
        count = 0
        while acc < 0.70:
            for i in trainingSet: #training the network
                net.back_propogate(i['attributes'], i['target'])
            acc = net.get_accuracy(validationSet)
            count += 1
            #print(str(count) + '|accuracy: ' + str(acc))
        print("Number of iterations: " + str(count))
        print("Accuracy: " + str(acc))
        print("")

    print('Finished')
    if recordTime:
        stop_time()
    print('accuracy training: ' + str(net.get_accuracy(trainingSet)))
    print('accuracy test: ' + str(net.get_accuracy(testingSet)))
    if verbose:
        net.output_nodes_and_weights()