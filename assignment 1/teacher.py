import random

MAX_GRID_COMBINATIONS = 19683 # this is technically more than needed
NUM_TEACHER_EXAMPLES = 350
LEARNING_RATE = 0.01
PLAYERX = 'X'
PLAYERO = 'O'
GRID_SIZE = 3

class Train_DATA_SET: # runs minmax algorithm
    def __init__(self, plyr):
        global MAX_GRID_COMBINATIONS, PLAYERX, PLAYERO
        self.player = plyr # 1 = p1, 2 = p2
        if plyr == 1:
            self.playerPiece = PLAYERX
            self.oppPiece = PLAYERO
        elif plyr == 2:
            self.playerPiece = PLAYERO
            self.oppPiece = PLAYERX
        else:
            print('Error [Train]: incorrect player piece')
            exit()
        
        # saving all move options to a hash value
        self.hash_answer = [[-1,0] for _ in range(MAX_GRID_COMBINATIONS)] # for each hash [move, move value]
        self.start_training_map()

    def has_winner(self, grid, lastMove):
        global GRID_SIZE
        x, y = lastMove
        plyr = grid[y][x]
        countx = 0
        county = 0
        cntdiag1 = 0
        cntdiag2 = 0
        for i in range(0,GRID_SIZE):
            if grid[i][x] == plyr:
                countx += 1
            if grid[y][i] == plyr:
                county += 1
        if x == y:
            for i in range(0,GRID_SIZE):
                if grid[i][i] == plyr:
                    cntdiag1 +=1
        if x == GRID_SIZE - y - 1:
            for i in range(0,GRID_SIZE):
                if grid[i][GRID_SIZE - i - 1] == plyr:
                    cntdiag2 += 1
        if countx == 3 or county == 3 or cntdiag1 == 3 or cntdiag2 == 3:
            return True
        return False
                
    def grid_hash(self, grid, plrPiece, oppPiece):
        total = 0
        multi = 0
        for r in grid:
            for c in r:
                val = 0
                if c == plrPiece: # player
                    val = 1
                elif c == oppPiece: # opponent
                    val = 2
                total += val * 3 ** multi
                multi += 1
        return total    

    def gen_training_values(self, grid, lastMove, maxf = True, depth = 0, newGenerate = True): #maxf : T = max, F = min
        hashVal = self.grid_hash(grid, self.playerPiece, self.oppPiece)
        if self.hash_answer[hashVal][0] != -1:  # check to see if we already found a value for this branch
            return self.hash_answer[hashVal][1]
        if lastMove[0] >= 0 and self.has_winner(grid, lastMove): # find winner. If not run next level of min max algorithm
            if maxf:
                self.hash_answer[hashVal] = [-2,-100] # -2 cause there is no move option for a winner
                return -100
            else:
                self.hash_answer[hashVal] = [-2,100] # -2 cause there is no move option for a winner
                return 100
        
        # running the min/max algorithm
        mVal = 0
        mMove = -2
        for y,r in enumerate(grid):
            for x,c in enumerate(r):
                if c == ' ':
                    if maxf:
                        grid[y][x] = self.playerPiece
                    else:
                        grid[y][x] = self.oppPiece
                    genVal = self.gen_training_values(grid, [x,y], (not maxf), depth + 1)
                    if mMove == -2:
                        mMove = 3*y+x
                        mVal = genVal
                    elif maxf and mVal < genVal: # for p1 we want to find the max()
                        mMove = 3*y+x
                        mVal = genVal
                    elif not maxf and mVal > genVal: #for p2 we want to find the min()
                        mMove = 3*y+x
                        mVal = genVal
                    grid[y][x] = ' '
        self.hash_answer[hashVal] = [mMove, mVal] # all possible values are saved in a grid
        return mVal

    def start_training_map(self):
        global GRID_SIZE
        grid = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.gen_training_values(grid, [-1,-1])

    def make_option(self, grid, player):
        if player == 1:
            hashVal = self.grid_hash(grid, self.oppPiece, self.playerPiece)
        else:
            hashVal = self.grid_hash(grid, self.playerPiece, self.oppPiece)
        moveVal = self.hash_answer[hashVal][0]
        if moveVal < 0:
            print(grid)
            print('error did not generate move for this grid')
            exit()
        y = int(moveVal / 3)
        x = moveVal % 3
        return x, y   
    def get_teacher_examples(self):
        examples = []
        board = []
        a = 0
        with open('training_examples.txt', 'r') as file:
            for line in file.readlines():
                if '[' in line:
                    board = []
                elif ':' in line:
                    astr = line.split(']')[0]
                    a = int(astr.replace(':','').strip())
                    examples.append([board, a])
                else:
                    board.append([int(x.strip()) for x in line.split()])
        return examples
                
    def generate_teacher_examples(self):
        global GRID_SIZE, NUM_TEACHER_EXAMPLES

        examples = []
        exampleCnt = 0

        draws = 0.40 * NUM_TEACHER_EXAMPLES
        interState = 0.40 * NUM_TEACHER_EXAMPLES
        endState = 0.20 * NUM_TEACHER_EXAMPLES
        beginingState = 0.40 * NUM_TEACHER_EXAMPLES

        while exampleCnt < NUM_TEACHER_EXAMPLES:
            hashNum = random.randint(1, MAX_GRID_COMBINATIONS - 1)
            if self.hash_answer[hashNum][0] == -1: # not a real grid
                continue
            Vtrain = self.hash_answer[hashNum][1]
            boardHash = [0 for _ in range(GRID_SIZE * GRID_SIZE)]
            cnt = 0
            p1cnt = 0
            p2cnt = 0
            empty = 0
            while hashNum > 0:
                piece = hashNum % 3
                if piece == 1:
                    p1cnt += 1
                elif piece == 2:
                    p2cnt += 1
                else:
                    empty += 1
                boardHash[cnt] = piece
                hashNum = int(hashNum/3)
                cnt +=1
            player = 1
            if p1cnt == p2cnt:
                player = 2
                Vtrain *= -1
                boardHash = [(x*2)%3 for x in boardHash] # quick fix to swap the 1 and 2s
            
            #checking distribution of examples
            if empty == 0:
                continue
            elif empty <= 3:
                if endState == 0:
                    continue
                if Vtrain == 0:
                    if draws == 0:
                        continue
                    draws -= 1
                endState -= 1
            elif empty >= 7:
                if beginingState == 0:
                    continue
                beginingState -= 1
            else:
                if interState == 0:
                    continue
                interState -= 1
            
            board = [ [0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            for i,v in enumerate(boardHash):
                board[int(i / 3)][int(i % 3)] = v

            examples.append([board, Vtrain])
            exampleCnt += 1
        return examples

    def get_test_board_states(self):
        return [
                [[2,0,0],
                [0,0,0],
                [0,0,1]],

                [[0,0,0],
                [0,2,0],
                [0,0,0]],

                [[2,0,0],
                [0,0,0],
                [0,0,0]],

                [[2,0,0],
                [0,1,0],
                [0,0,0]],

                [[0,0,0],
                [0,1,2],
                [0,0,0]],

                [[0,1,0],
                [0,2,0],
                [0,0,0]],

                [[0,0,0],
                [0,0,0],
                [0,0,0]],
        ]

if __name__ == '__main__':
    print("generating new training examples")
    TRAIN = Train_DATA_SET(1)
    examples = TRAIN.generate_teacher_examples()
    with open('training_examples.txt', 'w') as file:
        for grid, a in examples:
            file.write('[\n')
            for row in grid:
                file.write(" ".join([str(x) for x in row]) + '\n')
            file.write(':' + str(a) + '],\n')
    print("Finished generating examples")

