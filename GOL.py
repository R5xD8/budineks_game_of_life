import numpy as np
import numba as nb

@nb.jit(nopython = True, parallel=True)
def step_jit(board:nb.types.Array, next_board:nb.types.Array):
    nx, ny = board.shape
    cell_summ = 0

    for x in nb.prange(1, nx-1):
        for y in nb.prange (1, ny-1):
            cell_summ = np.sum(board[x-1:x+2, y-1:y+2]) - board[x,y]
            if cell_summ == 3:
                next_board[x,y] = 1
            elif cell_summ == 2:
                next_board[x,y] = board[x,y]
            else:
                next_board[x,y] = 0

    # sides
    next_board[0, 1:-1] = next_board[-2, 1:-1]
    next_board[-1, 1:-1] = next_board[1, 1:-1]
    next_board[1:-1, 0] = next_board[1:-1, -2]
    next_board[1:-1, -1] = next_board[1:-1, 1]
    # corners
    next_board[0,0] = next_board[-2,-2]
    next_board[0,-1] = next_board[-2,1]
    next_board[-1,0] = next_board[1, -2]
    next_board[-1,-1] = next_board[1,1]

    return next_board, board

class almost():

    def __init__(self,
                 number_of_colums = 100,
                 number_of_rows = 100
                 ):
        
        # create board w/ Halo
        self.board = np.zeros((number_of_rows + 2, number_of_colums + 2), dtype=np.int8)
        self.board[1:-1, 1:-1] = np.random.randint(0, 2, (number_of_rows, number_of_colums), dtype=np.int8)
        
        # create halo
        # sides
        self.board[0, 1:-1] = self.board[-2, 1:-1]
        self.board[-1, 1:-1] = self.board[1, 1:-1]
        self.board[1:-1, 0] = self.board[1:-1, -2]
        self.board[1:-1, -1] = self.board[1:-1, 1]
        # corners
        self.board[0,0] = self.board[-2,-2]
        self.board[0,-1] = self.board[-2,1]
        self.board[-1,0] = self.board[1, -2]
        self.board[-1,-1] = self.board[1,1]

        # create second board for step saves
        self.next_board = np.zeros_like(self.board, dtype=np.int8)

        # nb.config.THREADING_LAYER = "threadsafe"
    
    def step(self):

        self.board, self.next_board = step_jit(self.board, self.next_board)