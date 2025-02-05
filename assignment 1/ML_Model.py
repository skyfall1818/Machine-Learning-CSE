LEARNING_RATE = 0.01
GRID_SIZE = 3

class ML_Model:
    def __init__(self, weights = None):
        if weights is not None:
            self.variable_size = len(weights)
            self.MLWeights = weights
        else:
            n = 16
            self.variable_size = n
            self.MLWeights = [0.01 for _ in range(n)]

    def convert_board(self, board, player, opponent): # taking in the grid and creating a xi values
        global GRID_SIZE
        
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
                xi[y] = p1cnt
            else:
                xi[y] = -p2cnt
        
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
                xi[y+3] = p1cnt
            else:
                xi[y+3] = -p2cnt
        
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
            xi[6] = p1cnt 
        else:
            xi[6] = -p2cnt
        
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
            xi[7] = p1cnt
        else:
            xi[7] = -p2cnt
        return xi
        '''
    def train_set(self, trainSet):
        global GRID_SIZE, LEARNING_RATE
        prev = 0
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
                    vHatVal = sum([Xlist[ind] * self.MLWeights[ind] for ind in range(self.variable_size)])

                    # updating weight values
                    for i,weight in enumerate(self.MLWeights):
                        self.MLWeights[i] = weight + LEARNING_RATE * (a - vHatVal) * Xlist[i]
                    prev = vHatVal

    def train_non_teacher_set(self, moveList, win, first):
        global GRID_SIZE, LEARNING_RATE
        board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        trainSet = []
        p1 = 0
        p2 = 0
        a = 0

        if win:
            a = 100
        else:
            a = -100
        
        if first:
            p1 = 1
            p2 = 2
        else:
            p1 = 2
            p2 = 1
        
        player1 = True
        for x, y in moveList:
            if player1:
                board[x][y] = p1
                player1 = False
                if first:
                    trainSet.append([board, a])
            else:
                board[x][y] = p2
                player1 = True
                if not first:
                    trainSet.append([board, a])
        self.train_set(trainSet)

    def check_winner(self, grid, lastMove, plyr):
        global GRID_SIZE
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

    def make_option (self, board, player, opponent, null):
        global GRID_SIZE
        bestMove = [-1, -1]
        bestValue = 0
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if board[y][x] != null:
                    continue
                board[y][x] = player
                if self.check_winner(board, [x,y], player):
                    board[y][x] = null
                    return [x, y], 100
                bestOppValue = 0
                first = True
                # looks at opponent's possible move
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        if board[i][j] != null:
                            continue
                        board[i][j] = opponent
                        if self.check_winner(board, [j,i], opponent):
                            vHatVal = 100
                            board[i][j] = null
                        else:
                            xi = self.convert_board(board, opponent, player)
                            vHatVal = sum([xi[ind] * self.MLWeights[ind] for ind in range(self.variable_size)])
                        if first or bestOppValue < vHatVal:
                            first = False
                            bestOppValue = vHatVal
                        board[i][j] = null
                if bestMove[0] == -1 or bestOppValue < bestValue:
                    bestValue = bestOppValue
                    bestMove = [x, y]
                board[y][x] = null
        return bestMove, -bestValue
    
    def get_weights(self):
        return [x for x in self.MLWeights]