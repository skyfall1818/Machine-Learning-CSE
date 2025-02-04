#!/usr/bin/env python3 q
import random

GRID_SIZE = 3
MAX_GRID_COMBINATIONS = 19683 # this is technically more than needed
PLAYER1 = 'X'
PLAYER2 = 'O'
HUMAN = 'h'
TEACHER = 't'
MACHINE_LEARNER = 'm'
LEARNING_RATE = 0.001
NUM_TEACHER_EXAMPLES = 7000
NUM_INDIRECT_LEARNING = 100

class Hash_Node:
    def __init__(self):
        self.guess

class Train_DATA_SET: # runs minmax algorithm
    def __init__(self, plyr):
        global MAX_GRID_COMBINATIONS, PLAYER1, PLAYER2
        self.player = plyr # 1 = p1, 2 = p2
        if plyr == 1:
            self.playerPiece = PLAYER1
            self.oppPiece = PLAYER2
        elif plyr == 2:
            self.playerPiece = PLAYER2
            self.oppPiece = PLAYER1
        else:
            print('Error [Train]: incorrect player piece')
            exit()
        
        # saving all move options to a hash value
        self.hash_answer = [[-1,0] for _ in range(MAX_GRID_COMBINATIONS)] # for each hash [move, move value]
                        
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
                
    def grid_hash(self, grid):
        total = 0
        multi = 0
        for r in grid:
            for c in r:
                val = 0
                if c == self.playerPiece: # player
                    val = 1
                elif c == self.oppPiece: # opponent
                    val = 2
                total += val * 3 ** multi
                multi += 1
        return total    

    def gen_train_values(self, grid, lastMove, maxf = True, depth = 0): #maxf : T = max, F = min
        hashVal = self.grid_hash(grid)
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
                    genVal = self.gen_train_values(grid, [x,y], (not maxf), depth + 1)
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

    def start_traing_map(self):
        global GRID_SIZE
        grid = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.gen_train_values(grid, [-1,-1])

    def make_option(self, grid):
        hashVal = self.grid_hash(grid)
        moveVal = self.hash_answer[hashVal][0]
        if moveVal < 0:
            print('error did not generate move for this grid')
            exit()
        y = int(moveVal / 3)
        x = moveVal % 3
        return x, y   

    def gen_teacher_examples(self):
        global GRID_SIZE, NUM_TEACHER_EXAMPLES

        examples = []
        exampleCnt = 0

        draws = 0.45 * NUM_TEACHER_EXAMPLES
        interState = 0.40 * NUM_TEACHER_EXAMPLES
        endState = 0.40 * NUM_TEACHER_EXAMPLES
        beginingState = 0.20 * NUM_TEACHER_EXAMPLES

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
            if empty <= 3:
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
            

class ML_Model:
    def __init__(self, teacher = True, weights = None):
        if weights is not None:
            self.variable_size = len(weights)
            self.MLWeights = weights
            self.teacher = teacher
        else:
            n = 8
            self.variable_size = n
            self.MLWeights = [random.uniform(-1/n,1/n) for _ in range(n)]
            self.teacher = teacher

    def convert_board(self, board, player, opponent): # taking in the grid and creating a xi values
        global GRID_SIZE
        '''
        # [p1_row1, p1_row2, p1_row3, p1_col1, p1_col2, p1_col3, p1_diag1, p1_diag2, p2_row1, p2_row2, p2_row3, p2_col1, p2_col2, p2_col3, p2_diag1, p2_diag2]
        xi = [0 for _ in range(self.variable_size)]
        count = 0
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if board[y][x] == player:
                    xi[y] += 1
                elif board[y][x] == opponent:
                    xi[y+8] += 1
    
                if board[x][y] == player:
                    xi[3 + y] += 1
                elif board[x][y] == opponent:
                    xi[11 + y] += 1 # 3 + 8 = 11

            if board[y][y] == player:
                xi[6] += 1
            elif board[y][y] == opponent:
                xi[14] += 1
            
            if board[GRID_SIZE - y - 1][y] == player:
                xi[7] += 1
            elif board[GRID_SIZE - y - 1][y] == opponent:
                xi[15] += 1
        return xi

        '''
        # [row1, row2, row3, col1, col2, col3, diag1, diag2]
        xi = [0 for _ in range(self.variable_size)]

        for y in range(GRID_SIZE):
            p1cnt = 0
            p2cnt = 0
            for x in range(GRID_SIZE):
                if board[y][x] == player:
                    p1cnt += 1
                elif board[y][x] == opponent:
                    p2cnt += 1
            if p1cnt > 0 and p2cnt > 0:
                xi[y] = 0
            elif p1cnt > 0:
                xi[y] = p1cnt ** 2 + 1
            else:
                xi[y] = -(p2cnt ** 2)
        
        for y in range(GRID_SIZE):
            p1cnt = 0
            p2cnt = 0
            for x in range(GRID_SIZE):
                if board[x][y] == player:
                    p1cnt += 1
                elif board[x][y] == opponent:
                    p2cnt += 1
            if p1cnt > 0 and p2cnt > 0:
                xi[y+3] = 0
            elif p1cnt > 0:
                xi[y+3] = p1cnt ** 2 + 1
            else:
                xi[y+3] = -(p2cnt ** 2)
        
        p1cnt = 0
        p2cnt = 0
        for i in range(GRID_SIZE):
            if board[i][i] == player:
                p1cnt += 1
            elif board[i][i] == opponent:
                p2cnt += 1
        if p1cnt > 0 and p2cnt > 0:
            xi[6] = 0
        elif p1cnt > 0:
            xi[6] = p1cnt ** 2 + 1
        else:
            xi[6] = -(p2cnt ** 2)
        
        p1cnt = 0
        p2cnt = 0
        for i in range(GRID_SIZE):
            if board[GRID_SIZE - i - 1][i] == player:
                p1cnt += 1
            elif board[GRID_SIZE - i - 1][i] == opponent:
                p2cnt += 1
        if p1cnt > 0 and p2cnt > 0:
            xi[7] = 0
        elif p1cnt > 0:
            xi[7] = p1cnt ** 2 + 1
        else:
            xi[7] = -(p2cnt ** 2)
        return xi
        

    def train_teacher_set(self, trainSet):
        global GRID_SIZE, LEARNING_RATE
        for q, a in trainSet:
            for flip in [False, True]:
                b = [[c for c in r]  for r in q]
                if flip:
                    b = [ r[::-1]  for r in q]
                for _ in range(4):
                    #rotating 90 degrees
                    b = [ r[::-1]  for r in b]
                    for x in range(GRID_SIZE):
                        for y in range(GRID_SIZE):
                            b[x][y] = b [y][x]
                    Xlist = self.convert_board(b, 1, 2)
                    
                    # calculating v_hat
                    vHatVal = 0       
                    for i,weight in enumerate(self.MLWeights):
                        vHatVal += Xlist[i] * weight
                    
                    # updating weight values
                    for i,weight in enumerate(self.MLWeights):
                        self.MLWeights[i] = weight + LEARNING_RATE * (a - vHatVal) * Xlist[i]
        print('weights')
        print(self.MLWeights)

    def train_non_teacher_set(self, moveList, win, first, player_num):
        global GRID_SIZE, LEARNING_RATE
        board = [' ' for _ in range(GRID_SIZE * GRID_SIZE)]
        player = ''
        opp = ''
        if player_num == 1:
            player = PLAYER1
            opp = PLAYER2
        else:
            player = PLAYER2
            opp = PLAYER1
        a = 0
        if win:
            a = 100
        else:
            a = -100
        
        for move in moveList:
            if first:
                board[move] = player
                # taking in the grid and creating a xi values
                Xlist = self.convert_board(move, player, opp)
                
                # calculating v_hat
                vHatVal = 0
                for i,weight in enumerate(self.MLWeights):
                    vHatVal += Xlist[i] * weight
                
                # updating weight values
                for i,weight in enumerate(self.MLWeights):
                    self.MLWeights[i] = weight + LEARNING_RATE * (a - vHatVal) * Xlist[i]
            else:
                board[move] = opp

    def make_option (self, board, player, opponent):
        global GRID_SIZE
        bestMove = [-1, -1]
        bestValue = 0
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if board[y][x] != ' ':
                    continue
                board[y][x] = player
                bestOppValue = 0
                first = True
                # looks at opponent's possible move
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        if board[i][j] != ' ':
                            continue
                        board[i][j] = opponent
                        #print(board)
                        xi = self.convert_board(board, opponent, player)
                        #print(xi)
                        vHatVal = sum([xi[ind] * self.MLWeights[ind] for ind in range(self.variable_size)])
                        if first or vHatVal > bestOppValue:
                            first = False
                            bestOppValue = vHatVal
                        board[i][j] = ' '
                if bestMove[0] == -1 or bestOppValue < bestValue:
                    bestValue = bestOppValue
                    bestMove = [x, y]
                board[y][x] = ' '
        return bestMove

class Player_Manager:
    def __init__(self, player1, player2): # h = human, t = trainer, m = machine learner
        self.p1=player1
        self.p2=player2
    def make_move(self, grid, player):
        global TEACHER, HUMAN, MACHINE_LEARNER, PLAYER1, PLAYER2
        playermove = ''
        playertype = ''
        if player == 1:
            playertype = self.p1
        elif player == 2:
            playertype = self.p2          
        else:
            print('invalid player')
            exit()
        if playertype == HUMAN:
            return self.human_select()
        elif playertype == TEACHER:
            return TRAIN.make_option(grid)
        elif playertype == MACHINE_LEARNER:
            if player == 1:
                return ML.make_option(grid, PLAYER1, PLAYER2)
            else:
                return ML.make_option(grid, PLAYER2, PLAYER1)
        else:
            print('Player Manager Error: unknown player for ' + str(playertype))
                        
    def human_select(self):
        col = get_user_input('select col: ')
        row = get_user_input('select row: ')
        return col, row
        
class tictactoe_Game:
    def __init__(self):
        self.reset_grid()
        self.playerTurn=1
    
    def reset_grid(self):
        global GRID_SIZE
        self.grid =[ [ ' ' for _ in range(GRID_SIZE) ] for _ in range(GRID_SIZE) ]

    def play_turn(self):
        global PLAYER1, PLAYER2
        invalid = True
        failedCnt = 0
        while invalid:
            if failedCnt > 5:
                print('error selecting input. Timing out.')
                exit()
            x, y = PLAYER_MANAGER.make_move(self.grid, self.playerTurn)
            if x < 0 or x > GRID_SIZE or y < 0 or y > GRID_SIZE or self.grid[y][x] != ' ':
                get_user_raw_input('Invalid input for (' + str(x) + ',' + str(y) + '), try again...')
            else:
                invalid = False
            failedCnt += 1
        mv = PLAYER1
        if self.playerTurn == 2:
            mv = PLAYER2
        self.grid[y][x] = mv
        return self.check_winner([x,y], mv)

    def check_winner(self, lastMove, plyr):
        global GRID_SIZE
        x, y = lastMove
        countx = 0
        county = 0
        cntdiag1 = 0
        cntdiag2 = 0
        for i in range(0,GRID_SIZE):
            if self.grid[i][x] == plyr:
                countx += 1
            if self.grid[y][i] == plyr:
                county += 1
        if x == y:
            for i in range(0,GRID_SIZE):
                if self.grid[i][i] == plyr:
                    cntdiag1 +=1
        if x == GRID_SIZE - y - 1:
            for i in range(0,GRID_SIZE):
                if self.grid[i][GRID_SIZE - i - 1] == plyr:
                    cntdiag2 += 1
        if countx == 3 or county == 3 or cntdiag1 == 3 or cntdiag2 == 3:
            return True
        return False
            
    def print_grid(self):
        for r in self.grid:
            print( ' | '.join(r))
        print()
    def run_game(self):
        playAgain = True
        while playAgain:
            self.playerTurn = 1
            self.reset_grid()
            cnt = 0
            while cnt < 9:
                self.print_grid()
                winner = self.play_turn()
                if winner:
                    self.print_grid()
                    print('Player ' + str(self.playerTurn) + ' Wins!')
                    break
                self.playerTurn = (self.playerTurn%2) + 1
                cnt += 1
            if cnt == 9:
                self.print_grid()
                print('Tie')

            usrInput = -1
            while usrInput != 1 and usrInput != 2:
                print('Play again?   1: Yes | 2: No')
                usrInput = get_user_input ('> ')
                playAgain = (usrInput == 1)
def get_user_input(txt):
    return int(input(txt))
def get_user_raw_input(txt):
    return input(txt)


PLAYER_MANAGER= Player_Manager(MACHINE_LEARNER, HUMAN)
TRAIN = Train_DATA_SET(1)
ML = ML_Model()
#ML = ML_Model(True, [5,5,5,5,5,5,5,5])

if __name__ == '__main__':
    print('creating training map')
    TRAIN.start_traing_map()
    print('finished creating training map')
    print('generating training examples')
    ML.train_teacher_set(TRAIN.gen_teacher_examples())
    print('finish generating training examples')
    game = tictactoe_Game()
    game.run_game()
