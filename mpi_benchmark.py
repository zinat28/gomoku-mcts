import csv, os, statistics, sys
from mpi4py import MPI
from board import Board
from mcts import mcts
from mpi_mcts import mpi_mcts
from benchmark import POSITIONS, make_position   # reuse your positions

N_SIM = 10000
REPEATS = 5

def main():
    comm = MPI.COMM_WORLD
    rank, p = comm.Get_rank(), comm.Get_size()
    mpi_mcts(Board(), 200, 0, comm)              # warm-up, not timed
    rows = []
    for name, moves in POSITIONS.items():
        pos = make_position(moves)
        times = []
        for rep in range(REPEATS):
            comm.Barrier()
            t0 = MPI.Wtime()
            mpi_mcts(pos, N_SIM, rep, comm)
            comm.Barrier()
            times.append(MPI.Wtime() - t0)
        if rank == 0:
            mean, sd = statistics.mean(times), statistics.stdev(times)
            print(f"p={p} {name:6s} Tp={mean:7.3f}s ± {sd:.3f}  NPS={N_SIM/mean:9.0f}")
            rows.append([p, name, N_SIM, round(mean, 4), round(sd, 4), round(N_SIM / mean)])
    if rank == 0:
        os.makedirs("results", exist_ok=True)
        new = not os.path.exists("results/mpi.csv")
        with open("results/mpi.csv", "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["p", "position", "n_sim", "mean_time_s", "std_s", "nps"])
            w.writerows(rows)

if __name__ == "__main__":
    main()