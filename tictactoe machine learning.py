GRID_SIZE = 3
MAX_GRID_COMBINATIONS = 19683
PLAYER1 = 'X'
PLAYER2 = 'O'

class Hash_Node:
    def __init__(self):
        self.guess
class Train: # runs minmax algorithm
    def __init__(self, plyr):
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
        self.hash_answer = [[-1,0] for _ in range(19683)] # for each hash [move, move value]
                        
    def has_winner(self, grid, lastMove, plyr):
        x, y = lastMove
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
                elif c != '': # opponent
                    val = 2
                total += val + multi * 3
            multi += 1
        return total    
    def gen_train_values(self, grid, lastMove, maxf = True): #maxf : T = max, F = min
        # find winner. If not run next level of min max algorithm
        hashVal = grid_hash(grid) # check to see if we already found a value for this branch
        if self.hash_answer[hashVal][0] != -1:
            return self.hash_answer[hashVal][1]
        if self.has_winner(grid, lastMove):
            if not maxf:
                self.hash_answer[hashVal] = [-2,100] # -2 cause there is no move option for a winner
                return 100
            else:
                self.hash_answer[hashVal] = [-2,-100] # -2 cause there is no move option for a winner
                return -100
        mVal = 0
        mMove = -1
        for y,r in enumerate(grid):
            for x,c in enumerate(grid):
                if c == ' ':
                    if maxf:
                        grid[y][x] = self.playerPiece
                    else:
                        grid[y][x] = self.oppPiece
                    genVal = gen_train_value(grid, [x,y], (not maxf)))
                    grid[y][x] = ' '
                    if mMove = -1:
                        mMove = 3*y+x
                        mVal = genVal
                    elif maxf and mVal < genVal:
                        mMove = 3*y+x
                        mVal = genVal
                    elif not maxf and mVal > genVal:
                        mMove = 3*y+x
                        mVal = genVal 
        self.hash_answer[hashVal] = [mMove, mVal]
        return mVal

    def start_traing_map:
        grid = [[' ' for _ in range(GRID_SIZE] for _ in range(GRID_SIZE)]
        
    def make_option(self, grid):
        hashVal = grid_hash(grid)
        moveVal = self.hash_answer[hashVal]
        y = int(moveVal/3)
        x = moveVal%3
        return x, y


TRAIN = Train(1)
class Player_Manager:
    def __init__(self, player1='h', player2='t'): # h = human, t = trainer, m = machine learner
        self.p1=player1
        self.p2=player2
    def make_move(self, grid, player):
        playermove = ''
        playertype = ''
        if player == 1:
            playertype = self.p1
        elif player == 2:
            playertype = self.p2          
        else:
            print('invalid player')
            exit()
        if playertype == 'h':
            return self.human_select()
        else: # need logic for this
            return TRAIN.make_option(grid)
                        
    def human_select(self):
        col = get_user_input('select col: ')
        row = get_user_input('select row: ')
        return col, row
        
class tictactoe_Game:
    def __init__(self):
        self.grid =[ [ ' ' for _ in range(GRID_SIZE) ] for _ in range(GRID_SIZE) ]
        self.playerTurn=1
    def play_turn(self):
        x, y = PLAYER_MANAGER.make_move(self.grid, self.playerTurn)
        mv = PLAYER1
        if self.playerTurn == 2:
            mv = PLAYER2
        self.grid[y][x] = mv
        return self.check_winner([x,y], mv)
    def check_winner(self, lastMove, plyr):
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
        while True:
            self.print_grid()
            winner = self.play_turn()
            if winner:
                self.print_grid()
                print('player ' + str(self.playerTurn) + ' wins!')
                break
            self.playerTurn = (self.playerTurn%2) + 1

def get_user_input(txt):
    return int(input(txt))
    
PLAYER_MANAGER= Player_Manager()

if __name__ == '__main__':
    print('creating training map')
    TRAIN.gen_train_values
    print('finished creating training map')
    game = tictactoe_Game()
    game.run_game()
