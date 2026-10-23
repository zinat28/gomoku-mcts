from multiprocessing import Pool
from board import Board, SIZE
from mp_mcts import mp_mcts

def idx(r, c): return r * SIZE + c

if __name__ == "__main__":
    b = Board()
    for bm, wm in [((7,3),(0,0)), ((7,4),(0,2)), ((7,5),(0,4)), ((7,6),(0,6))]:
        b.play(idx(*bm)); b.play(idx(*wm))
    with Pool(4) as pool:
        move = mp_mcts(b, 5000, 0, pool, 4)
    assert move in (idx(7, 2), idx(7, 7)), divmod(move, SIZE)
    print("multiprocessing test passed")