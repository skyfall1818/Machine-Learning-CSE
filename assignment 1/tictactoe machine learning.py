#!/usr/bin/env python3
from teacher import Train_DATA_SET
from ML_Model import ML_Model

GRID_SIZE = 3
MAX_GRID_COMBINATIONS = 19683 # this is technically more than needed
PLAYERX = 'X'
PLAYERO = 'O'
HUMAN = 'h'
TEACHER = 't'
MACHINE_LEARNER_TEACHER = 'mt'
MACHINE_LEARNER_NO_TEACHER = 'm'
NUM_INDIRECT_LEARNING = 25

class Player_Manager:
    def __init__(self, player1, player2, player1Icon, player2Icon): # h = human, t = trainer, m = machine learner
        self.p1=player1
        self.p2=player2
        self.p1Icon = player1Icon
        self.p2Icon = player2Icon

    def make_move(self, grid, player):
        global TEACHER, HUMAN, TRAIN, MACHINE_LEARNER_TEACHER, MACHINE_LEARNER_NO_TEACHER, MACHINE_LEARNER_NO_TEACHER, PLAYERX, PLAYERO
        playermove = ''
        playertype = ''
        icon = ''
        oppIcon = ''
        if player == 1:
            playertype = self.p1
            icon = self.p1Icon
            oppIcon = self.p2Icon
        elif player == 2:
            playertype = self.p2
            icon = self.p2Icon  
            oppIcon = self.p1Icon   
        else:
            print('invalid player')
            exit()
        if playertype == HUMAN:
            return self.human_select(), icon
        elif playertype == TEACHER:
            return TRAIN.make_option(grid, player), icon

        elif playertype == MACHINE_LEARNER_TEACHER:
            out = ML_TEACHER.make_option(grid, icon, oppIcon, ' ')
            return out[0], icon
            
        elif playertype == MACHINE_LEARNER_NO_TEACHER:
            out = ML.make_option(grid, icon, oppIcon, ' ')
            return out[0], icon
        else:
            print('Player Manager Error: unknown player for ' + str(playertype))
            exit()
                        
    def human_select(self):
        col = get_user_input('select col: ')
        row = get_user_input('select row: ')
        return col, row
        
class tictactoe_Game:
    def __init__(self, output = True):
        global GRID_SIZE
        self.grid =[ [ ' ' for _ in range(GRID_SIZE) ] for _ in range(GRID_SIZE) ]
        self.moveList = []
        self.winner = 0
        self.playerTurn=1
        self.output = output
        self.cnt = 0

    def set_grid(self, grid, pT):
        self.grid = grid
        self.cnt = 0
        for r in grid:
            for c in r:
                if c != ' ':
                    self.cnt +=1
        self.playerTurn = pT
     
    def print_grid(self):
        print('  0 1 2')
        for i,r in enumerate(self.grid):
            print( str(i) + ' ' + '|'.join(r))
            if i < GRID_SIZE -1:
                print( '  -+-+-')
        print()

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
    
    def play_turn(self):
        invalid = True
        failedCnt = 0
        cord = []
        while invalid:
            if failedCnt > 5:
                print('error selecting input. Timing out.')
                exit()

            cord, icon = PLAYER_MANAGER.make_move(self.grid, self.playerTurn)
            x, y = cord
            if x < 0 or x > GRID_SIZE or y < 0 or y > GRID_SIZE or self.grid[y][x] != ' ':
                print(self.grid)
                get_user_raw_input('Invalid input for (' + str(x) + ',' + str(y) + '), try again...')
            else:
                invalid = False
            failedCnt += 1
        self.moveList.append(cord)
        self.grid[y][x] = icon
        return self.check_winner([x,y], icon)
    
    def run_game(self):
        cnt = self.cnt
        while cnt < 9:
            if self.output:
                self.print_grid()
            winner = self.play_turn()
            if winner:
                if self.output:
                    self.print_grid()
                    print('Player ' + str(self.playerTurn) + ' Wins!')
                self.winner = self.playerTurn
                break
            self.playerTurn = (self.playerTurn%2) + 1
            cnt += 1
        if cnt == 9 and self.output:
            self.print_grid()
            print('Tie')
    def get_winner(self):
        return self.winner
    def move_list(self):
        return [ x for x in self.moveList]

def get_user_input(txt):
    return int(input(txt))

def get_user_raw_input(txt):
    return input(txt)

def print_weights():
    print('with teacher weights')
    print([round(x, 1) for x in ML_TEACHER.get_weights()])
    print('without teacher weights')
    print([round(x, 1) for x in ML.get_weights()])

def generate_weights():
    global ML_TEACHER, TRAIN, ML, PLAYER_MANAGER, NUM_INDIRECT_LEARNING, MACHINE_LEARNER_NO_TEACHER, PLAYERX, PLAYERO
    ML_TEACHER = ML_Model()
    ML = ML_Model()
    print('generating training examples')
    ML_TEACHER.train_set(TRAIN.get_teacher_examples())
    print('finish generating training examples')
    print('ML inderect training by playing itself')
    PLAYER_MANAGER = Player_Manager(MACHINE_LEARNER_NO_TEACHER, MACHINE_LEARNER_NO_TEACHER, PLAYERX, PLAYERO)
    for _ in range(NUM_INDIRECT_LEARNING):
        trainingGame = tictactoe_Game(False)
        trainingGame.run_game()
        ML.train_non_teacher_set(trainingGame.move_list(), trainingGame.get_winner() == 1, True)
        ML.train_non_teacher_set(trainingGame.move_list(), trainingGame.get_winner() == 2, False)
    print('finish indirect training')

def test_models():
    global GRID_SIZE, TRAIN, PLAYERX, PLAYERO, PLAYER_MANAGER
    boards = TRAIN.get_test_board_states()
    t_icon = PLAYERO
    m_icon = PLAYERX

    wins_t = 0
    wins_nt = 0
    loss_t = 0
    loss_nt = 0
    ties_t = 0
    ties_nt = 0

    for grid in boards:
        outputBoard = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        cnt1 = 0
        cnt2 = 0
        first = True

        for y,r in enumerate(grid):
            for x,c in enumerate(r):
                if c == 1:
                    cnt1 += 1
                    outputBoard[y][x] = m_icon
                if c == 2:
                    cnt2 += 1
                    outputBoard[y][x] = t_icon

        game = tictactoe_Game(False)
        currentPlayer = 0
        if cnt1 == cnt2:
            PLAYER_MANAGER = Player_Manager(MACHINE_LEARNER_TEACHER, TEACHER, m_icon, t_icon)
            game.set_grid(outputBoard, 1)
            currentPlayer = 1
        else:
            PLAYER_MANAGER = Player_Manager(TEACHER, MACHINE_LEARNER_TEACHER, t_icon, m_icon)
            game.set_grid(outputBoard, 2)
            currentPlayer = 2
        game.run_game()
        winner = game.get_winner()
        if winner == 0:
            ties_t += 1
        elif winner == currentPlayer:
            wins_t += 1
        else:
            loss_t += 1
        
        outputBoard = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for y,r in enumerate(grid):
            for x,c in enumerate(r):
                if c == 1:
                    outputBoard[y][x] = m_icon
                if c == 2:
                    outputBoard[y][x] = t_icon
        game = tictactoe_Game(False)
        currentPlayer = 0
        if cnt1 == cnt2:
            PLAYER_MANAGER = Player_Manager(MACHINE_LEARNER_NO_TEACHER, TEACHER, m_icon, t_icon)
            game.set_grid(outputBoard, 1)
            currentPlayer = 1
        else:
            PLAYER_MANAGER = Player_Manager(TEACHER, MACHINE_LEARNER_NO_TEACHER, t_icon, m_icon)
            game.set_grid(outputBoard, 2)
            currentPlayer = 2
        game.run_game()
        winner = game.get_winner()
        if winner == 0:
            ties_nt += 1
        elif winner == currentPlayer:
            wins_nt += 1
        else:
            loss_nt += 1
    print('Teacher Model had Win: ' + str(wins_t) + ', Ties: ' +  str(ties_t) + ', Loss: ' + str(loss_t))
    print('No Teacher Model had Win: ' + str(wins_nt) + ', Ties: ' +  str(ties_nt) + ', Loss: ' + str(loss_nt))
        

PLAYER_MANAGER = Player_Manager(HUMAN, TEACHER, PLAYERX, PLAYERO)
TRAIN = Train_DATA_SET(1)

ML_TEACHER = ML_Model()
ML =         ML_Model()

if __name__ == '__main__':
    generate_weights()

    menuInput = 0
    AIopponent = ''
    while True:
        print('Choose Option: 1-Play Against ML_Model | 2-Play Against Teacher | 3-Test ML Model | 4-Print Weights | 5-Quit')
        menuInput = get_user_input('> ')
        if menuInput == 5:
            break
        elif menuInput == 1:
            print('Choose Opponent: 1-ML Model with teacher | 2-ML Model without teacher')
            oppInput = get_user_input('> ')
            if oppInput == 1:
                AIopponent = MACHINE_LEARNER_TEACHER
            elif oppInput == 2:
                AIopponent = MACHINE_LEARNER_NO_TEACHER
            else:
                get_user_raw_input("error: invalid input try again")
                continue
        elif menuInput == 2:
            AIopponent = TEACHER
        elif menuInput == 3:
            test_models()
            continue
        elif menuInput == 4:
            print_weights()
            continue
        else:
            get_user_raw_input("error: invalid input try again")
            continue
        
        playAgain = True
        while playAgain:

            #selecting to go first or second
            usrInput = -1
            while usrInput not in [1,2]:
                print('Do you want to start first?   1-yes | 2-no')
                usrInput = get_user_input ('> ')
                if usrInput ==1 or usrInput==2:
                    playAgain = (usrInput == 1)
                else:
                    get_user_input ('invalid input. try again.')
            if usrInput == 1:
                PLAYER_MANAGER= Player_Manager(HUMAN, AIopponent, PLAYERX, PLAYERO)
            else:
                PLAYER_MANAGER= Player_Manager(AIopponent, HUMAN, PLAYERO, PLAYERX)
            
            game = tictactoe_Game()
            game.run_game()
            usrInput = -1

            #selecting replay game
            while usrInput not in [1,2]:
                print('Play again?   1: Yes | 2: No')
                usrInput = get_user_input ('> ')
                if usrInput ==1 or usrInput==2:
                    playAgain = (usrInput == 1)
                else:
                    get_user_input ('invalid input. try again.')
