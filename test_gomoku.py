from board import Board, SIZE
from mcts import mcts

def idx(r, c): return r * SIZE + c

def test_win_detection():
    b = Board()
    for k in range(5):
        won = b.play(idx(7, 3 + k))      # Black row
        if k < 4:
            b.play(idx(0, k * 2))        # White scattered
    assert won, "five in a row should be detected"

def test_finds_win_in_one():
    b = Board()
    # Black has four in a row (7,3)-(7,6); White stones are harmless
    for bm, wm in [((7,3),(0,0)), ((7,4),(0,2)), ((7,5),(0,4)), ((7,6),(0,6))]:
        b.play(idx(*bm)); b.play(idx(*wm))
    move, _ = mcts(b, 5000, seed=0)
    assert move in (idx(7, 2), idx(7, 7)), f"expected winning move, got {divmod(move, SIZE)}"

def test_deterministic():
    b = Board()
    assert mcts(b, 500, seed=42)[0] == mcts(b, 500, seed=42)[0]

if __name__ == "__main__":
    test_win_detection(); test_finds_win_in_one(); test_deterministic()
    print("All tests passed.")