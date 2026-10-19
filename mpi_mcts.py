from mpi4py import MPI
from mcts import mcts


def mpi_mcts(board, n_sim, base_seed, comm):
    """Root parallelization: every rank searches its own tree, then visits are summed."""
    rank, size = comm.Get_rank(), comm.Get_size()

    # split the budget: n_sim = sum of local budgets
    local = n_sim // size + (1 if rank < n_sim % size else 0)

    visits = {}
    if local > 0:
        _, visits = mcts(board, local, seed=base_seed + rank)   # distinct seed per rank

    all_visits = comm.gather(visits, root=0)     # collect dicts on rank 0

    best = None
    if rank == 0:
        total = {}
        for v in all_visits:
            for move, n in v.items():
                total[move] = total.get(move, 0) + n   # REDUCE: sum root-child visits
        best = max(total, key=total.get)

    return comm.bcast(best, root=0)              # everybody learns the chosen move