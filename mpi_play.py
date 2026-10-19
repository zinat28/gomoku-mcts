from mpi4py import MPI
from board import Board, SIZE
from mpi_mcts import mpi_mcts

N_SIM = 5000


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    b = Board()
    if rank == 0:
        print(f"Running on {comm.Get_size()} MPI ranks.")
        print("You are X (first). Enter 'row col', e.g. 7 7")

    while True:
        move = None
        if rank == 0:
            print(b)
            while True:
                try:
                    r, c = map(int, input("Your move: ").split())
                    m = r * SIZE + c
                    if not (0 <= r < SIZE and 0 <= c < SIZE) or b.cells[m] != 0:
                        print("Invalid move.")
                        continue
                    move = m
                    break
                except ValueError:
                    print("Type two numbers.")
        move = comm.bcast(move, root=0)          # all ranks get the human move

        if b.play(move):
            if rank == 0:
                print(b); print("You win!")
            return
        if b.is_full():
            if rank == 0:
                print("Draw.")
            return

        ai = mpi_mcts(b, N_SIM, 1, comm)         # collective: all ranks must call it
        if rank == 0:
            print(f"AI plays {divmod(ai, SIZE)}")
        if b.play(ai):
            if rank == 0:
                print(b); print("AI wins!")
            return


if __name__ == "__main__":
    main()