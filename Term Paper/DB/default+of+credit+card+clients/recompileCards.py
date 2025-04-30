import random
random.seed(0)
FILE = 'data.txt'
COMPILE_FILE = 'CreditCards'
PERCENTAGE_TRAIN = 0.8

ATTRIBUTES = []

TARGET = [ 'Y', 'N']

DATA = []
TRAIN = []
TEST = []

with open(FILE, 'r') as f:
    for i,line in enumerate(f.readlines()):
        if i == 0:
            continue
        if i == 1:
            ATTRIBUTES = line.split()[1:-1]
            print(ATTRIBUTES)
            continue
        line = line[:-1]
        DATA.append({'Values': line.split()[1:-1], 'Target': TARGET[int(line.split()[-1])]})

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