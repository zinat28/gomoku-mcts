SIZE = 15
N = SIZE * SIZE
EMPTY, BLACK, WHITE = 0,1,2
DIRS = ((0,1),(1,0),(1,1),(1,-1))

class Board:
    __slots__=("cells","player", "last", "moves_played")

    def __init__(self):
        self.cells= bytearray(N)
        self.player = BLACK
        self.last= -1
        self.moves_played = 0

    def copy(self):
        b = Board.__new__(Board)
        b.cells = bytearray(self.cells)
        b.player = self.player
        b.last = self.last
        b.moves_played= self.moves_played
        return b

    def legal_moves(self):
        return[i for i in range(N) if self.cells[i]==EMPTY]
    
    def is_full(self):
        return self.moves_played==N
    
    def play(self, move):
        self.cells[move]= self.player
        self.last = move 
        self.moves_played+=1
        won = self.check_win(move)
        self.player = 3 - self.player 
        return won

    def check_win(self,move):
        color = self.cells[move]
        r,c = divmod(move, SIZE)
        for dr, dc in DIRS:
            count = 1
            for sign in (1,-1):
                rr, cc = r+dr*sign , c+dc *sign
                while 0 <= rr <SIZE and 0<= cc<SIZE and self.cells[rr*SIZE +cc] ==color:
                    count +=1
                    rr += dr* sign
                    cc += dc* sign
                if count >=5:
                    return True
            return False
    def __str__(self):
        sym = {EMPTY: " .", BLACK: " x", WHITE: "0"}
        header = "  "+" ".join(f"{c:2d}" for c in range(SIZE))
        rows = [f"{r:2d}" + " ".join(f"{sym[self.cells[r*SIZE+c]]}"for c in range(SIZE))
                for r in range(SIZE)]
        return header + "\n" + "\n".join(rows)