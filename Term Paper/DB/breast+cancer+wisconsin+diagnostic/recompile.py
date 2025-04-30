import random
random.seed("recompile")
FILE = 'wdbc.data'
COMPILE_FILE = 'BrestCancer'
PERCENTAGE_TRAIN = 0.8

ATTRIBUTES = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension',
            'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
            'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2']

TARGET = ['M', 'B']

DATA = []
TRAIN = []
TEST = []

with open(FILE, 'r') as f:
    for line in f:
        line = line[:-1]
        DATA.append({'Values': line.split(',')[2:], 'Target': line.split(',')[1]})

random.shuffle(DATA)

length = int(len(DATA)*PERCENTAGE_TRAIN)
TRAIN = DATA[:length]
TEST = DATA[length:]

with open(COMPILE_FILE + '-attr.txt', 'w') as f:
    for attr in ATTRIBUTES:
        f.write(attr + ' continuous\n')
    f.write('\n' + ' '.join(TARGET) + '\n')

with open(COMPILE_FILE + '-train.txt', 'w') as f:
    for line in TRAIN:
        f.write(' '.join(line['Values']) + ' ' + line['Target'] + '\n')

with open(COMPILE_FILE + '-test.txt', 'w') as f:
    for line in TEST:
        f.write(' '.join(line['Values']) + ' ' + line['Target'] + '\n')
