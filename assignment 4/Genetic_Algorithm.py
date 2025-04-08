import sys
import random
import math

POPULATION_SIZE = 100
REPLACEMENT_RATE = 0.7
MUTATION_RATE = 0.1
ITERATIONS = 100

BINARY = True
ATTRIBUTE_LENGTH = 0
ATTRIBUTES = {}
TARGET = {}

class Genetic_Algorithm:
    def __init__(self, populationSize, replacementRate, mutationRate, iterations, attributes, target, attributeLength, trainset, binary= True):
        self.populationSize = populationSize
        self.replacementRate = replacementRate
        self.mutationRate = mutationRate
        self.iterations = iterations
        self.population = []
        self.fitnessScores = []
        self.attributes = attributes
        self.target = target
        self.attributeLength = attributeLength
        self.trainSet = trainset
        self.binary = binary
        self.initialize_population()
    
    # individuals are set by [[arttribute], default value]
    def initialize_population(self):
        for _ in range(self.populationSize):
            # create a random individual
            if self.binary == False:
                individual = [random.uniform(0, 5) for _ in range(self.attributeLength)]
                individual.append(random.randint(0, len(self.target) - 1))
                individual.append(random.randint(0, len(self.target) - 1)) # another copy to include base case
            else:
                individual = [random.randint(0, 1) for _ in range(self.attributeLength)]
                individual.append(random.randint(0, len(self.target) - 1))
                individual.append(random.randint(0, len(self.target) - 1)) # another copy to include base case
            self.population.append(individual)
            self.fitnessScores.append(0)
            self.calclate_fitness_scores_and_heap(self.trainSet)

    def learn(self, trainSet, iteration):
        for i in range(iteration):
            # select parents for the next generation
            parents, parent_fitness = self.select_parents()
            children = parents

            fitness = parent_fitness
            fitness_total = sum(fitness)

            # create children by crossover
            while len(children) < self.populationSize:
                rng = random.uniform(0, fitness_total)
                index = 0
                while rng > 0:
                    rng -= fitness[index]
                    index += 1
                parent1 = parents[index - 1]
                rng = random.uniform(0, fitness_total)
                index = 0
                while rng > 0:
                    rng -= fitness[index]
                    index += 1
                parent2 = parents[index - 1]
                # crossover function
                children.extend(self.crossover(parent1, parent2))
                fitness.extend([0, 0])
            if len(children) > self.populationSize: # remove extra children
                children = children[:self.populationSize]
                fitness = fitness[:self.populationSize]
            
            # set the new population
            self.fitnessScores = fitness
            self.population = children

            # add mutations
            self.add_mutation()

            # calculate the fitness scores
            self.calclate_fitness_scores_and_heap(trainSet)
            print("iteration", i, "best fitness score:", self.population[0], self.fitnessScores[0])


    # select surviving parents from the population to move on to the next generation
    def select_parents(self):
        surviors = self.populationSize - int(self.populationSize * self.replacementRate)
        parents = []
        parent_fitness = []
        total = sum(self.fitnessScores)
        for _ in range(surviors):
            num_gen = random.uniform(0, total)
            index = 0
            while num_gen > 0:
                num_gen -= self.fitnessScores[index]
                index += 1 # at the end, we move 1 index too far. Will be subtracted later
            parents.append(self.population[index - 1])
            parent_fitness.append(self.fitnessScores[index - 1])
            # heapify the list
            prntInd = len(parents) - 1
            while prntInd > 0 and self.fitnessScores[prntInd] > self.fitnessScores[(prntInd - 1) // 2]:
                self.fitnessScores[prntInd], self.fitnessScores[(prntInd - 1) // 2] = self.fitnessScores[(prntInd - 1) // 2], self.fitnessScores[prntInd]
                self.population[prntInd], self.population[(prntInd - 1) // 2] = self.population[(prntInd - 1) // 2], self.population[prntInd]
                prntInd = (prntInd - 1) // 2
        return parents, parent_fitness

    # using the GABIL algorithm
    def crossover(self, parent1, parent2):
        children = []

        if random.random() < self.mutationRate:
            children.append(parent1[:-1] + parent2[:-1] + parent1[-1:])
            children.append(parent2[:-1] + parent1[:-1] + parent2[-1:])
            return children
        
        randomIndex = [random.randint(0, self.attributeLength) for _ in range(2)]
        left = min(randomIndex)
        d2 = self.attributeLength - max(randomIndex)

        rightP1 = len(parent1) - d2
        rightP2 = len(parent2) - d2

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
            if random.random() < self.mutationRate:
                self.population[index] = individual[:-1] + individual[:-1] + individual[-1:]
                self.fitnessScores[index] = 0
                break
            mutate = False
            for i in range(len(individual)):
                if random.random() < self.mutationRate:
                    mutate = True
                    if self.binary:
                        individual[i] = (individual[i] + 1) % 2
                        if self.test_empty_hypothesis(individual, i):
                            individual[i] = (individual[i] + 1) % 2
                            mutate = False
                    else:
                        if i % (self.attributeLength + 1) == self.attributeLength or i == len(individual) - 1:
                            again = True
                            while again:
                                temp = individual[i]
                                individual[i] = random.randint(0, len(self.target))
                                if temp != individual[i]:
                                    again = False
                        else:
                            cnt = 0
                            for _ in range(10):
                                cnt += 1
                                individual[i] += random.uniform(-2.5, 2.5)
                                if not self.test_empty_hypothesis(individual, i):
                                    break
                            if cnt == 10:
                                mutate = False
            if mutate:
                self.fitnessScores[index] = 0

    # test the hypothesis of and attributes of an empty set
    def test_empty_hypothesis(self, individual, index = None): # False if there is no empty hypothsis, True if there is an empty hypothesis
        if index is None:
            if self.binary:
                index = 0
                while len(individual) - index > self.attributeLength + 1:
                    for att in self.attributes:
                        attLen = len(att)
                        null = True
                        for i in range(attLen):
                            if individual[i + index] == 1:
                                null = False
                                break
                        if null:
                            return True
                    index += attLen + 1
                return False
            else:
                for i in range(0, len(individual) - 1, 2):
                    if individual[i] >= individual[i + 1]:
                        return True
                return False
        else:
            if index == len(individual) - 1:
                return False
            if self.binary:
                attIndex = 0
                i = 0
                attributeList = list(self.attributes.keys())
                while i < index:
                    if attIndex == 0 and i != 0:
                        i += 1
                    i += len(self.attributes[attributeList[attIndex]])
                    attIndex = (attIndex + 1) % len(self.attributes)
                attIndex = (attIndex - 1) % len(self.attributes)
                for j in range(len(self.attributes[attributeList[attIndex]])):
                    try:
                        if individual[i - j - 1] == 1:
                            return False
                    except IndexError:
                        print(index, i, j)
                        print(individual)
                        exit(-1)
                return True
            else:
                if index % 2 == 0:
                    if individual[index] < individual[index + 1]:
                        return False
                    return True
                else:
                    if individual[index] > individual[index - 1]:
                        return False
                    return True

    def calclate_fitness_scores_and_heap(self, trainSet): # heaping allows us to keep track of the best individual
        for heapIndex, individual in enumerate(self.population):
            if self.fitnessScores[heapIndex] != 0: # skipping individuals with fitness scores already calculated
                continue
            self.fitnessScores[heapIndex] = self.calculate_fitness(individual, trainSet)
            index = heapIndex
            while index > 0 and self.fitnessScores[index] > self.fitnessScores[(index - 1) // 2]:
                # swap the current index with its parent
                self.fitnessScores[index], self.fitnessScores[(index - 1) // 2] = self.fitnessScores[(index - 1) // 2], self.fitnessScores[index]
                self.population[index], self.population[(index - 1) // 2] = self.population[(index - 1) // 2], self.population[index]
                index = (index - 1) // 2

    # calculate the fitness of an individual
    def calculate_fitness(self, individual, trainSet):
        num_correct = 0
        for data in trainSet:
            attrubute = data['attributes']
            target = data['target']
            out = self.get_output(attrubute, individual)
            if out == target:
                num_correct += 1
        return num_correct / len(trainSet)

    def get_accuracy(self, testSet):
        correct = 0
        for data in testSet:
            out = self.get_output(data['attributes'], self.population[0])
            if out == data['target']:
                correct += 1
        return correct / len(testSet)
    
    def get_output(self, instance, individual):
        indexcnt = 0
        while len(individual) - indexcnt > self.attributeLength + 1:
            if self.binary:
                found = True
                for i in range(len(instance)):
                    if instance[i] == 1:
                        if individual[indexcnt + i] != 1:
                            found = False
                            break
                if found:
                    return individual[indexcnt + len(instance)]
                indexcnt += len(instance) + 1
            else:
                i = 0
                found = True
                for instanceVal in instance:
                    if instanceVal < individual[indexcnt] or instanceVal > individual[indexcnt + i + 1]:
                        found = False
                        break
                    i += 2
                if found:
                    return individual[indexcnt + self.attributeLength]
                indexcnt += self.attributeLength + 1
        return individual[-1]
    
    # print the population
    def print_population(self, top=None):
        if top is None:
            for i,individual in enumerate(self.population):
                print(individual, self.fitnessScores[i])
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

def read_settings_file(file):
    global POPULATION_SIZE, REPLACEMENT_RATE, MUTATION_RATE, NUM_ITERATIONS
    with open(file, "r") as f:
        for line in f.readlines():
            if '=' not in line:
                continue
            name,val = line.split("=")
            val  = val.strip()
            if 'Population' in name:
                POPULATION_SIZE = int(val)
            elif 'Replacement' in name:
                REPLACEMENT_RATE = float(val)
            elif 'Mutation' in name:
                MUTATION_RATE = float(val)
            elif 'Iterations' in name:
                NUM_ITERATIONS = int(val)

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
                if att[1] == 'continuous':
                    BINARY = False
                    cnt += 1
            else: # saves the target
                if first and len(lines) == i + 1:
                    TARGET = [i.strip() for i in line.split()[1:]]
                else:
                    first = False
                    TARGET.append( [i.strip() for i in line.split()[1:]])
    ATTRIBUTE_LENGTH = cnt

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
            data['target'] = [ i for i in range(len(TARGET)) if TARGET[i] in inputs[-1]][0]
            dataList.append(data)
    return dataList

def main():
    if len(sys.argv) <= 1:
        print("No arguments provided.")
        exit(-1)
    
    #if len(sys.argv) < 5:
    #    print("Invalid number of arguments.")
    #    exit(-1)

    settingsFile = sys.argv[1] #argument 1
    read_settings_file(settingsFile)
    print(POPULATION_SIZE, REPLACEMENT_RATE, MUTATION_RATE, NUM_ITERATIONS)

    attributeFile = sys.argv[2] #argument 2
    read_attributes_file(attributeFile)
    print(ATTRIBUTE_LENGTH, ATTRIBUTES, TARGET)

    trainFile = sys.argv[3] #argument 3
    trainingSet = read_data_file(trainFile)
    print(trainingSet)

    testFile = sys.argv[4] #argument 4
    testingSet = read_data_file(testFile)
    print(testingSet)

    ga = Genetic_Algorithm(POPULATION_SIZE, REPLACEMENT_RATE, MUTATION_RATE, NUM_ITERATIONS, ATTRIBUTES, TARGET, ATTRIBUTE_LENGTH, trainingSet, BINARY)
    ga.print_population()
    ga.learn(trainingSet, NUM_ITERATIONS)
    print("Accuracy: ", ga.get_accuracy(testingSet))

if __name__ == "__main__":
    main()