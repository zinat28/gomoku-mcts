SIZE = 15
N = SIZE * SIZE
EMPTY, BLACK, WHITE = 0, 1, 2
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Board:
    __slots__ = ("cells", "player", "last", "moves_played")

    def __init__(self):
        self.cells = bytearray(N)
        self.player = BLACK
        self.last = -1
        self.moves_played = 0

    def copy(self):
        b = Board.__new__(Board)
        b.cells = bytearray(self.cells)
        b.player = self.player
        b.last = self.last
        b.moves_played = self.moves_played
        return b

    def legal_moves(self):
        return [i for i in range(N) if self.cells[i] == EMPTY]

    def candidate_moves(self):
        """Empty cells within 2 steps of any stone (centre if board is empty)."""        
        if self.moves_played == 0:
            return [N // 2]
        res = set()
        for i in range(N):
            if self.cells[i] != EMPTY:
                r, c = divmod(i, SIZE)
                for rr in range(max(0, r - 2), min(SIZE, r + 3)):
                    for cc in range(max(0, c - 2), min(SIZE, c + 3)):
                        j = rr * SIZE + cc
                        if self.cells[j] == EMPTY:
                            res.add(j)
        return sorted(res)

    def is_full(self):
        return self.moves_played == N

    def play(self, move):
        self.cells[move] = self.player
        self.last = move
        self.moves_played += 1
        won = self.check_win(move)
        self.player = 3 - self.player
        return won

    def check_win(self, move):
        color = self.cells[move]
        r, c = divmod(move, SIZE)
        for dr, dc in DIRS:
            count = 1
            for sign in (1, -1):
                rr, cc = r + dr * sign, c + dc * sign
                while 0 <= rr < SIZE and 0 <= cc < SIZE and self.cells[rr * SIZE + cc] == color:
                    count += 1
                    rr += dr * sign
                    cc += dc * sign
            if count >= 5:
                return True
        return False

    def __str__(self):
        sym = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        header = "   " + "".join(f"{c:<3d}" for c in range(SIZE))
        rows = []
        for r in range(SIZE):
            cells = "".join(f"{sym[self.cells[r * SIZE + c]]:<3}" for c in range(SIZE))
            rows.append(f"{r:<3d}" + cells)
        return header + "\n" + "\n".join(rows)