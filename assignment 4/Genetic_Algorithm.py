import sys
import random
import math

POPULATION_SIZE = 100 # population size of each generation
REPLACEMENT_RATE = 0.7 # probability of survival
MUTATION_RATE = 0.01 # probability of a mutation
EXPANDING_MUTATION_RATE = 0.0001 # adding a mutation to the end to increase size of the individual
ITERATIONS = 100 # number of iterations

BINARY = True # descrete or continuous attributes
ATTRIBUTE_LENGTH = 0 # number of attributes
ATTRIBUTES = {} # dictionary of attributes
TARGET = [] # list of the target values

# STRATEGIES: fitness, tournament, rank, all
STRAT_DICT = {'F': 'Fitness', 'T': 'Tournament', 'R': 'Rank'}
STRATEGIES = ['F', 'T', 'R']
STRATEGY = 'F'

# Genetic Algorithm
# this aglorithm hold the machine learning algorithm and the knowledge
# inputs: populationSize, replacementRate, mutationRate, expandingMutationRate, iterations, attributes, target, attributeLength
# individuals are set by [[arttribute], default value]
#   default values are for the algorithm if none of the rules match the example
# continuous attributes uses two values:
#   first is the point value
#   second is the range
class Genetic_Algorithm:
    def __init__(self, populationSize, replacementRate, mutationRate, expandingMutationRate, iterations, attributes, target, attributeLength, strategy, trainSet, binary= True):
        self.populationSize = populationSize
        self.replacementRate = replacementRate
        self.mutationRate = mutationRate
        self.expandingMutationRate = expandingMutationRate
        self.iterations = iterations
        self.population = [] # list of individuals
        self.fitnessScores = [] # list of fitness scores for each individual in the population by index
        self.attributes = attributes
        self.target = target
        self.attributeLength = attributeLength
        self.strategy = strategy  # strategy of selection of parents
        self.binary = binary # discrete or continuous attributes
        self.initialize_population(trainSet) # initialize the population

        # using linear rank
        # normaly, this equation should have the numerator be (2 * (populationSize - i + 1)), but the heap sort will sort the list from worst to best, making the last element the highest rank
        self.rank_values = [(2 * (i + 1))/(populationSize * (populationSize + 1)) for i in range(populationSize)]

    # initialize the population by setting random values
    def initialize_population(self, trainSet):
        for _ in range(self.populationSize):
            # create a random individual
            if self.binary == False:
                individual = []
                for _ in range(len(self.target)-1): # we cant to scale the values based on the number of target values, so the algori
                    individual.extend([random.uniform(0, 5) if i %2 == 1 else random.uniform(-5, 5) for i in range(self.attributeLength)])
                    individual.append(random.randint(0, len(self.target) - 1))
                individual.append(random.randint(0, len(self.target) - 1)) # another copy to include base case
            else:
                individual = [random.randint(0, 1) for _ in range(self.attributeLength)]
                individual.append(random.randint(0, len(self.target) - 1))
                individual.append(random.randint(0, len(self.target) - 1)) # another copy to include base case
            self.population.append(individual)
            self.fitnessScores.append(0)
            self.calclate_fitness_scores_and_heap(trainSet)

    # runs the genetic algorithm
    def learn(self, trainSet, iteration, output=False):
        for i in range(iteration):
            # select parents for the next generation
            parents, parent_fitness = self.select_parents()
            children = parents

            fitness = parent_fitness
            fitness_total = sum(fitness)
            rank_total = len(parents) * (len(parents) + 1)

            # create children by crossover
            while len(children) < self.populationSize:
                # select parent1 by using the fitness scores of the survivors
                if self.strategy == "F" or self.strategy == "R":
                    rng = random.uniform(0, fitness_total)
                    index = 0
                    while rng > 0:
                        rng -= fitness[index]
                        index += 1
                    parent1 = parents[index - 1] # we overshoot the index by 1 so we subtract it

                    # select parent2 by using the fitness scores of the survivors
                    rng = random.uniform(0, fitness_total)
                    index2 = 0
                    while rng > 0:
                        rng -= fitness[index2]
                        index2 += 1
                    parent2 = parents[index2 - 1] # we overshoot the index by 1 so we subtract it
                    # crossover function
                    children.extend(self.crossover(parent1, parent2))
                    fitness.extend([0, 0])
                elif self.strategy == "T":
                    winners = []
                    for _ in range(2):
                        i1 = random.randint(0, len(parents) - 1)
                        i2 = random.randint(0, len(parents) - 1)
                        winnerIndex = -1
                        if self.fitnessScores[i1] > self.fitnessScores[i2]:
                            winnerIndex = i1
                        else:
                            winnerIndex = i2
                        winners.append(parents[winnerIndex])
                    children.extend(self.crossover(winners[0], winners[1]))
                    fitness.extend([0, 0])

            if len(children) > self.populationSize: # remove extra children
                children = children[:self.populationSize]
                fitness = fitness[:self.populationSize]
            
            # set the new population
            self.fitnessScores = fitness
            self.population = children

            # add mutations to the population
            self.add_mutation()

            # calculate the fitness scores
            self.calclate_fitness_scores_and_heap(trainSet)

            if output: # prints out the best individual each iteration
                if self.binary:
                    print("iteration", i, "best fitness score:", self.population[0], self.fitnessScores[0])
                else:
                    print("iteration", i, "best fitness score:", ['{:.2f}'.format(i) for i in self.population[0]], self.fitnessScores[0])


    # select surviving parents from the population to move on to the next generation
    def select_parents(self):
        surviors = self.populationSize - int(self.populationSize * self.replacementRate)
        parents = []
        parent_fitness = []
        total = sum(self.fitnessScores) # we uses the total fitness and skip normalizing the function
        for popcnt in range(surviors):
            # select the parent by using one of the strategies
            if self.strategy == "F": # rank selection
                num_gen = random.uniform(0, total) # random number between 0 and the total
                # search the population for the parent based on fitness
                index = 0
                while num_gen > 0:
                    num_gen -= self.fitnessScores[index]
                    index += 1
                index -= 1 # at the end, we move the index 1 too far. will neeed to subtract 1
                parents.append(self.population[index])
                parent_fitness.append(self.fitnessScores[index])
            elif self.strategy == "R": # rank selection
                self.sort_population()
                num_gen = random.random() # random number between 0 and 1
                # search the population for the parent based on rank
                index = 0
                while num_gen > 0:
                    num_gen -= self.rank_values[index]
                    index += 1 
                index -= 1 # at the end, we move the index 1 too far. will neeed to subtract 1
                parents.append(self.population[index])
                parent_fitness.append(self.fitnessScores[index])
            elif self.strategy == "T": # tournament selection
                index1 = random.randint(0, self.populationSize - 1)
                index2 = random.randint(0, self.populationSize - 1)
                winnerIndex = -1
                if self.fitnessScores[index1] > self.fitnessScores[index2]:
                    winnerIndex = index1
                else:
                    winnerIndex = index2
                parents.append(self.population[winnerIndex])
                parent_fitness.append(self.fitnessScores[winnerIndex])

            # heapify the list
            searchInd = popcnt
            while searchInd > 0 and parent_fitness[searchInd] > parent_fitness[(searchInd - 1) // 2]:
                # swap the current index with its parent
                parent_fitness[searchInd], parent_fitness[(searchInd - 1) // 2] = parent_fitness[(searchInd - 1) // 2], parent_fitness[searchInd]
                parents[searchInd], parents[(searchInd - 1) // 2] = parents[(searchInd - 1) // 2], parents[searchInd]
                searchInd = (searchInd - 1) // 2 # move to parent
        return parents, parent_fitness

    # sorts the population
    # since the population data structure is a heap, we can sort by the pop head algorithm
    # sorts the population from worst to best
    def sort_population(self):
        for i in range(self.populationSize):
            lastIndex = self.populationSize - i - 1
            self.population[0], self.population[lastIndex] = self.population[lastIndex], self.population[0]
            self.fitnessScores[0], self.fitnessScores[lastIndex] = self.fitnessScores[lastIndex], self.fitnessScores[0]
            index = 0
            while index * 2 + 1 < lastIndex:
                if self.fitnessScores[index * 2 + 1] > self.fitnessScores[(index * 2) + 2]:
                    if self.fitnessScores[index] >= self.fitnessScores[index * 2 + 1]:
                        break
                    self.fitnessScores[index], self.fitnessScores[index * 2 + 1] = self.fitnessScores[index * 2 + 1], self.fitnessScores[index]
                    self.population[index], self.population[index * 2 + 1] = self.population[index * 2 + 1], self.population[index]
                    index = index * 2 + 1
                else:
                    if self.fitnessScores[index] >= self.fitnessScores[(index * 2) + 2]:
                        break
                    self.fitnessScores[index], self.fitnessScores[(index * 2) + 2] = self.fitnessScores[(index * 2) + 2], self.fitnessScores[index]
                    self.population[index], self.population[(index * 2) + 2] = self.population[(index * 2) + 2], self.population[index]
                    index = (index * 2) + 2

    # using the GABIL crossover algorithm
    def crossover(self, parent1, parent2):
        children = []
        
        # takes 2 random numbers, the highest number will be the d2, the lowest will be d1
        randomIndex = [random.randint(0, self.attributeLength) for _ in range(2)]
        left = min(randomIndex) # getting d1
        d2 = self.attributeLength - max(randomIndex)

        # getting the indeces for the right side for each parent
        rightP1 = len(parent1) - d2
        rightP2 = len(parent2) - d2

        # main cross over agorithm

        # swap the first half of parent1, middle of parent2, and last half of parent1
        child = parent1[:left] + parent2[left:rightP2] + parent1[rightP1:]
        if not self.test_empty_hypothesis(child):
            children.append(child)

        # swap the first half of parent2, middle of parent1, and last half of parent2
        child = parent2[:left] + parent1[left:rightP1] + parent2[rightP2:]
        if not self.test_empty_hypothesis(child):
            children.append(child)
        return children
    
    # add mutaitons to the population
    def add_mutation(self):
        for index, individual in enumerate(self.population):

            # check to see if the individual would mutate by expanding
            if random.random() < self.expandingMutationRate:
                if len(individual) // (self.attributeLength + 1) > len(self.target) and random.random() < 0.5: 
                    lengthindex = len(individual) - self.attributeLength
                    self.population[index] = individual[lengthindex:-1] + individual[:lengthindex] + individual[-1:]
                    self.fitnessScores[index] = 0
                else: # add new random rule
                    randExtend = []
                    if self.binary:
                        randExtend = [random.randint(0, 1) for _ in range(self.attributeLength)]
                    else:
                        randExtend = [random.uniform(0, 5) if i % 2 == 1 else random.uniform(-5, 5) for i in range(self.attributeLength)]
                    randExtend.append(random.randint(0, len(self.target) - 1))
                    self.population[index] = individual[:-1] + randExtend + individual[-1:]
                    self.fitnessScores[index] = 0
                continue

            mutate = False
            for i in range(len(individual)):
                if random.random() < self.mutationRate:
                    mutate = True
                    if self.binary: # mutating a descrete variable by swapping the bit
                        individual[i] = (individual[i] + 1) % 2
                        if self.test_empty_hypothesis(individual, i): # double check that the hypothesis is still valid
                            individual[i] = (individual[i] + 1) % 2
                            mutate = False
                    else: # mutating a continuous variable by adding a random number
                        if i % (self.attributeLength + 1) == self.attributeLength or i == len(individual) - 1: # mutating the target values which are descrete
                            again = True
                            while again:
                                temp = individual[i]
                                individual[i] = random.randint(0, len(self.target)-1)
                                if temp != individual[i]:
                                    again = False
                        else: # mutating the continuous variables
                            temp = individual[i]
                            cnt = 0
                            for _ in range(10):
                                individual[i] += random.uniform(-2.5, 2.5)
                                if not self.test_empty_hypothesis(individual, i): # double check that the hypothesis is still valid
                                    break
                                cnt += 1
                            if cnt == 10:
                                individual[i] = temp
                                mutate = False
            if mutate: # if the individual mutated, recalculate its fitness score
                self.fitnessScores[index] = 0

    # test the hypothesis of and attributes of an empty set
    # False if there is no empty hypothsis, True if there is an empty hypothesis
    def test_empty_hypothesis(self, individual, index = None):
        # indexing lets us test a specific attribute rather than the entire hypothesis
        if index is None: # look at the entire hypothesis
            if self.binary:
                index = 0
                # groups eacha attribute together and see if the values are all 0
                while len(individual) - index > self.attributeLength + 1: # this make sure we check the next rule
                    for att in self.attributes:
                        attLen = len(att)
                        null = True
                        for i in range(attLen):
                            if individual[i + index] == 1: # we have a 1 so this hypothesis is valid
                                null = False
                                break
                        if null:
                            return True
                    index += attLen + 1
                return False
            
            else:
                # check the range of each attribute to make sure it is > 0       
                for i in range((len(individual) - 1) // (self.attributeLength + 1)):
                    for j in range(1, self.attributeLength, 2):
                        if individual[i * (self.attributeLength + 1) + j] < 0:
                            return True
                return False
        else: # given a specific index
            if index == len(individual) - 1 or index % (self.attributeLength + 1) == self.attributeLength: # ignore the target values
                return False
            
            if self.binary:
                # search for the attribute and indeces of the attribute
                attIndex = 0
                i = 0
                attributeList = list(self.attributes.keys())
                while i < index: # goes through each attribute until it finds the index
                    if attIndex == 0 and i != 0:
                        i += 1
                    i += len(self.attributes[attributeList[attIndex]])
                    attIndex = (attIndex + 1) % len(self.attributes)
                attIndex = (attIndex - 1) % len(self.attributes) # go to the first index of the attribute

                # cheack each value of the attribute and check for the empty set
                for j in range(len(self.attributes[attributeList[attIndex]])):
                    if individual[i - j - 1] == 1:
                        return False
                return True
            else:
                if (index % (self.attributeLength + 1)) % 2 == 1: # checks the range
                    if individual[index] > 0: # check if the range is positive
                        return False
                    return True
                else:
                    return False

    # fitness is calulated based on the accuracy of the hypothesis to the training set
    # heaping allows us to keep track of the best individual
    def calclate_fitness_scores_and_heap(self, trainSet):
        
        for heapIndex, individual in enumerate(self.population):

            # skipping individuals with fitness scores already calculated
            if self.fitnessScores[heapIndex] != 0:
                continue

            # calculate the fitness score
            self.fitnessScores[heapIndex] = self.calculate_fitness(individual, trainSet)
            
            # add the individual to the heap
            index = heapIndex
            while index > 0 and self.fitnessScores[index] > self.fitnessScores[(index - 1) // 2]:
                # swap the current index with its parent
                self.fitnessScores[index], self.fitnessScores[(index - 1) // 2] = self.fitnessScores[(index - 1) // 2], self.fitnessScores[index]
                self.population[index], self.population[(index - 1) // 2] = self.population[(index - 1) // 2], self.population[index]
                index = (index - 1) // 2 # move to parent

    # calculate the fitness of an individual
    def calculate_fitness(self, individual, trainSet):
        num_correct = 0
        # go through the training set
        for data in trainSet:
            attrubute = data['attributes']
            # compare the output
            out = self.get_output(attrubute, individual)
            if out == data['target']:
                num_correct += 1
        # return the accuracy
        return num_correct / len(trainSet)

    # calculate the accuracy of the best individual
    def get_accuracy(self, testSet):
        correct = 0
        # go through the training set
        for data in testSet:
            # compare the output
            out = self.get_output(data['attributes'], self.population[0])
            if out == data['target']:
                correct += 1
        # return the accuracy
        return correct / len(testSet)
    
    # gets the guessed output for the individual based on the example
    def get_output(self, instance, individual):
        indexcnt = 0
        # searches through each rule
        while len(individual) - indexcnt > self.attributeLength + 1:
            if self.binary: # descrete
                found = True
                for i in range(len(instance)):
                    # we only check if the attribute from the example is 1
                    if instance[i] == 1:
                        if individual[indexcnt + i] != 1: # if the rule does not match, move on to the next rule
                            found = False
                            break
                if found: # found the rule that matches the example
                    return individual[indexcnt + len(instance)]
            else: # continuous
                found = True
                # check each attribute in the examples
                for i, instanceVal in enumerate(instance):
                    point = individual[indexcnt + i * 2 + 1]
                    pointRange = individual[indexcnt + i * 2]
                    upperBound = point + pointRange
                    lowerBound = point - pointRange
                    # check if the value is within the range
                    if instanceVal <= lowerBound or instanceVal >= upperBound:
                        found = False
                        break
                if found: # found the rule that matches the example
                    return individual[indexcnt + self.attributeLength]
            indexcnt += self.attributeLength + 1 # move the index count to the next rule
        # if we find no rule that matches, we take the default target value
        return individual[-1]
    
    def print_best_rule(self):
        individual = self.population[0]
        rules = [individual[i: i + self.attributeLength + 1] for i in range(0, len(individual[:-1]), self.attributeLength + 1)]
        attrList = list(self.attributes.keys())
        if self.binary:
            for i,rule in enumerate(rules):
                txt = 'rule ' + str(i + 1) + ': '
                index = 0
                attIndex = 0
                conjuction = []
                while index + 1 < self.attributeLength:
                    disjunction = []
                    for att in self.attributes[attrList[attIndex]]:
                        if rule[index] == 1:
                            disjunction.append(att)
                        index += 1
                    if disjunction:
                        conjuction.append( attrList[attIndex] + ' = ' + ' or '.join(disjunction))
                    attIndex += 1
                txt += ', '.join(conjuction) + ' => ' + self.target[rule[-1]]
                print(txt)
        else:
            for i,rule in enumerate(rules):
                txt = 'rule ' + str(i + 1) + ': '
                conjuction = []
                for i, att in enumerate(self.attributes):
                    upper = rule[i * 2] + rule[i * 2 + 1]
                    lower = rule[i * 2] - rule[i * 2 + 1]
                    conjuction.append(attrList[i] + ' = [' + '{:0.2f}'.format(lower) + ' : ' + '{:0.2f}'.format(upper) + ']')
                txt += ', '.join(conjuction) + ' => ' + self.target[rule[-1]]
                print(txt)
        print('else:', self.target[individual[-1]])

            
    # print the population
    # this is used for debugging
    def print_population(self, top=None):
        if top is None:
            for i,individual in enumerate(self.population):
                print(['{:0.2f}'.format(score) for score in individual], self.fitnessScores[i])
        else:
            save = []
            for _ in range(top):
                bestscore = 0
                bestindex = 0
                for i, ind in enumerate(self.population):
                    if i in save:
                        continue
                    if self.fitnessScores[self.population.index(ind)] > bestscore:
                        bestscore = self.fitnessScores[self.population.index(ind)]
                        bestindex = self.population.index(ind)
                save.append(bestindex)
                print(self.population[bestindex], self.fitnessScores[bestindex])

# reads the settings file
# checks for the key words
# Note: the order of the key words is not important
def read_settings_file(file):
    global POPULATION_SIZE, REPLACEMENT_RATE, MUTATION_RATE, NUM_ITERATIONS, EXPANDING_MUTATION_RATE, STRAT
    with open(file, "r") as f:
        # reads each line
        for line in f.readlines():
            # skips lines that are not settings lists
            if '=' not in line or '#' in line:
                continue
            name,val = line.split("=") # look for the key '=' to separate the name from the value
            val  = val.strip()

            # gets the values associated with the name
            if 'Population' in name:
                POPULATION_SIZE = int(val)
            elif 'Replacement' in name:
                REPLACEMENT_RATE = float(val)
            elif 'Mutation' in name:
                MUTATION_RATE = float(val)
            elif 'Iterations' in name:
                NUM_ITERATIONS = int(val)
            elif 'Expanding' in name:
                EXPANDING_MUTATION_RATE = float(val)
            elif 'Strategy' in name: # gets the strategy
                if val == 'Fitness':
                    STRAT = 'F'
                elif val == 'Rank':
                    STRAT = 'R'
                elif val == 'Tournament':
                    STRAT = 'T'

            else: # unknown setting``
                print("Unknown setting: " + name)
                exit(-1)

# reads the attributes file save it the the global variable ATTRIBUTES and TARGETS
def read_attributes_file(file):
    global ATTRIBUTE_LENGTH, ATTRIBUTES, TARGET, BINARY
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
                cnt += len(att) - 1

                ATTRIBUTES[att[0]] = att[1:]
                if att[1] == 'continuous': # for continuous inputs we will add a new attribute value
                    BINARY = False
                    cnt += 1
            else: # saves the target
                TARGET = [i.strip() for i in line.split()[1:]]
    ATTRIBUTE_LENGTH = cnt

# reads the traninig set and test set files
# targets and attributes are staored based on how it will be implemented in the network
def read_data_file(file):
    global ATTRIBUTES
    dataList = []
    attList = list(ATTRIBUTES.keys())

    # opening file
    with open(file, "r") as f:
        for line in f.readlines():
            # data file
            data = {}
            instance = []
            inputs = line.split()

            # saving data and target values
            for i, val in enumerate(inputs):
                # ignore the last value since it is the target
                if i+1 == len(inputs):
                    break
                elif len (ATTRIBUTES[attList[i]]) == 1: # for continuous inputs
                    instance.append(float(val))
                else: # for discrete inputs
                    attVals = [ 0 for _ in range(len(ATTRIBUTES[attList[i]]))]
                    for i, item in enumerate(ATTRIBUTES[attList[i]]):
                        if item in val: # matching attribute values
                            attVals[i] = 1
                            break
                    instance.extend(attVals)

            # saving data inside a dictionary
            data['attributes'] = instance
            data['target'] = [ i for i in range(len(TARGET)) if TARGET[i] in inputs[-1]][0]
            dataList.append(data)
    return dataList

def main():
    global POPULATION_SIZE, REPLACEMENT_RATE, MUTATION_RATE, NUM_ITERATIONS, EXPANDING_MUTATION_RATE, ATTRIBUTES, TARGET, ATTRIBUTE_LENGTH, STRATEGY, STRATEGIES, BINARY
    random.seed('Assignment 4')
    if len(sys.argv) <= 1:
        print("No arguments provided.")
        exit(-1)
    
    if len(sys.argv) < 5:
        print("Invalid number of arguments.")
        exit(-1)

    settingsFile = sys.argv[1] # argument 1
    read_settings_file(settingsFile)

    attributeFile = sys.argv[2] # argument 2
    read_attributes_file(attributeFile)

    trainFile = sys.argv[3] # argument 3
    trainingSet = read_data_file(trainFile)

    testFile = sys.argv[4] # argument 4
    testingSet = read_data_file(testFile)

    outputBest = False
    varyReplacement = False
    varyGeneration = False
    # read any additional arguments
    # note: will ignore any arguments that are not valid past the 4th
    index = 5
    while index < len(sys.argv):
        if sys.argv[index] == '-v' or sys.argv[index] == '--verbose': # verbose
            outputBest = True
        elif sys.argv[index] == '-r' or sys.argv[index] == '--replacement': # replacement
            varyReplacement = True
        elif sys.argv[index] == '-g' or sys.argv[index] == '--generation': # generation
            varyGeneration = True
        index += 1

    if not varyGeneration and not varyReplacement: 
        # run the genetic algorithm
        ga = Genetic_Algorithm(POPULATION_SIZE, REPLACEMENT_RATE, MUTATION_RATE, EXPANDING_MUTATION_RATE, NUM_ITERATIONS, ATTRIBUTES, TARGET, ATTRIBUTE_LENGTH, STRATEGY, trainingSet, BINARY)
        print('Learning...')

        # tranining the GA
        ga.learn(trainingSet, NUM_ITERATIONS)

        # output the results based on the best individual
        print("Accuracy of best on training set: ", ga.get_accuracy(trainingSet))
        print("Accuracy of best on test set: ", ga.get_accuracy(testingSet))
        if outputBest: # otuput the best rule in readable form
            ga.print_best_rule()
    else:
        for strategy in STRATEGIES: # run through all the strategies
            print('-----Strategy: ' + STRAT_DICT[strategy] + '-----')
            if varyGeneration: # vary the number of generations. Output the results
                ga = Genetic_Algorithm(POPULATION_SIZE, REPLACEMENT_RATE, MUTATION_RATE, EXPANDING_MUTATION_RATE, NUM_ITERATIONS, ATTRIBUTES, TARGET, ATTRIBUTE_LENGTH, strategy, trainingSet, BINARY)
                INCR = int(NUM_ITERATIONS * 0.1)
                iterations = 0
                for _ in range(0, NUM_ITERATIONS, INCR):
                    iterations += INCR
                    print('Generations: ' + str(iterations))
                    ga.learn(trainingSet, INCR)
                    print("Accuracy of best on training set: ", ga.get_accuracy(trainingSet))
                    print("Accuracy of best on test set: ", ga.get_accuracy(testingSet))
                    if outputBest:
                        ga.print_best_rule()
                    print()
            else: # vary the replacement rate. Output the results
                replacements = [i/10 for i in range(0,10)]
                for replacement in replacements:
                    ga = Genetic_Algorithm(POPULATION_SIZE, replacement, MUTATION_RATE, EXPANDING_MUTATION_RATE, NUM_ITERATIONS, ATTRIBUTES, TARGET, ATTRIBUTE_LENGTH, strategy, trainingSet, BINARY)
                    print('Replacement_Rate: ' + str(replacement))
                    # tranining the GA
                    ga.learn(trainingSet, NUM_ITERATIONS)

                    # output the results based on the best individual
                    print("Accuracy of best on training set: ", ga.get_accuracy(trainingSet))    
                    print("Accuracy of best on test set: ", ga.get_accuracy(testingSet))
                    if outputBest:
                        ga.print_best_rule()
                    print()

if __name__ == "__main__":
    main()