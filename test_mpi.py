from mpi4py import MPI
from board import Board, SIZE
from mpi_mcts import mpi_mcts

def idx(r, c): return r * SIZE + c

comm = MPI.COMM_WORLD
b = Board()
for bm, wm in [((7,3),(0,0)), ((7,4),(0,2)), ((7,5),(0,4)), ((7,6),(0,6))]:
    b.play(idx(*bm)); b.play(idx(*wm))

move = mpi_mcts(b, 5000, 0, comm)
if comm.Get_rank() == 0:
    assert move in (idx(7, 2), idx(7, 7)), f"got {divmod(move, SIZE)}"
    print("MPI test passed on", comm.Get_size(), "ranks")