import numpy as np
import numba as nb

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# Parallel Halo

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

@nb.jit(nopython = True, parallel=True)
def step_par_hal(board:nb.types.Array, next_board:nb.types.Array):
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

class GOL_with_JIT_par_hal():

    def __init__(self, board):
        
        # create board w/ Halo
        self.board = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
        self.board[1:-1, 1:-1] = np.array(board, dtype=np.int8)
        
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
    
    def step(self):

        self.board, self.next_board = step_par_hal(self.board, self.next_board)

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# Non-parallel Halo

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

@nb.jit(nopython = True)
def step_hal(board:nb.types.Array, next_board:nb.types.Array):
    nx, ny = board.shape
    cell_summ = 0

    for x in range(1, nx-1):
        for y in range (1, ny-1):
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

class GOL_with_JIT_hal():

    def __init__(self, board):
        
        # create board w/ Halo
        self.board = np.zeros((board.shape[0] + 2, board.shape[1] + 2), dtype=np.int8)
        self.board[1:-1, 1:-1] = np.array(board, dtype=np.int8)
        
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
    
    def step(self):

        self.board, self.next_board = step_hal(self.board, self.next_board)

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# Parallel modulo

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
@nb.jit(nopython = True, parallel=True)
def step_par_mod(board:nb.types.Array, next_board:nb.types.Array):
    nx, ny = board.shape

    for x in nb.prange(nx):
        for y in nb.prange (ny):
            cell_summ = - board[x, y]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cell_summ += board[(x+dx)%nx, (y+dy)%ny]

            if cell_summ == 3:
                next_board[x,y] = 1
            elif cell_summ == 2:
                next_board[x,y] = board[x,y]
            else:
                next_board[x,y] = 0

    return next_board, board

class GOL_with_JIT_par_mod():

    def __init__(self, board):
        
        # create board
        self.board = np.array(board, dtype=np.int8)
        
        # allocate next board
        self.next_board = np.zeros_like(self.board, dtype=np.int8)
    
    def step(self):

        self.board, self.next_board = step_par_mod(self.board, self.next_board)

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# Non-parallel modulo

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

@nb.jit(nopython = True)
def step_mod(board:nb.types.Array, next_board:nb.types.Array):
    nx, ny = board.shape

    for x in range(nx):
        for y in range (ny):
            cell_summ = - board[x, y]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cell_summ += board[(x+dx)%nx, (y+dy)%ny]

            if cell_summ == 3:
                next_board[x,y] = 1
            elif cell_summ == 2:
                next_board[x,y] = board[x,y]
            else:
                next_board[x,y] = 0

    return next_board, board

class GOL_with_JIT_mod():

    def __init__(self, board):
        
         # create board
        self.board = np.array(board, dtype=np.int8)
        
        # allocate next board
        self.next_board = np.zeros_like(self.board, dtype=np.int8)
    
    def step(self):

        self.board, self.next_board = step_mod(self.board, self.next_board)