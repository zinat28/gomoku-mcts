from board import Board, SIZE
from mcts import mcts

N_SIM = 5000

def main():
    b = Board()
    print("You are X (first). Enter 'row col', e.g. 7 7")
    while True:
        print(b)
        try:
            r, c = map(int, input("Your move: ").split())
            m = r * SIZE + c
            if not (0 <= r < SIZE and 0 <= c < SIZE) or b.cells[m] != 0:
                print("Invalid move."); continue
        except ValueError:
            print("Type two numbers."); continue
        if b.play(m):
            print(b); print("You win!"); return
        if b.is_full():
            print("Draw."); return
        move, _ = mcts(b, N_SIM, seed=1)
        print(f"AI plays {divmod(move, SIZE)}")
        if b.play(move):
            print(b); print("AI wins!"); return

if __name__ == "__main__":
    main()