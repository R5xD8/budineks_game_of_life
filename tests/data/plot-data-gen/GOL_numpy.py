import numpy as np

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# Halo

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

class GOL_just_numpy_hal:
    def __init__(self, board):
        
        nx, ny = board.shape

        # create board w/ Halo
        self.board = np.zeros((nx + 2, ny + 2), dtype=np.int8)
        self.board[1:-1, 1:-1] = np.random.randint(0, 2, (nx, ny), dtype=np.int8)
        
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
        board = self.board
        next_board = self.next_board

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

        self.board = next_board
        self.next_board = board
        

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# No Halo

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

class GOL_just_numpy_mod:
    def __init__(self, board):
        
        nx, ny = board.shape

        # create board w/ Halo
        self.board = board

        # create second board for step saves
        self.next_board = np.zeros_like(self.board, dtype=self.board.dtype)

    
    def step(self):
        board = self.board
        next_board = self.next_board

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
        
        self.board = next_board
        self.next_board = board