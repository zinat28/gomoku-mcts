import csv, os, statistics, time
from multiprocessing import Pool
from board import Board
from mp_mcts import mp_mcts
from benchmark import POSITIONS, make_position

N_SIM = 10000
REPEATS = 5
PROCS = [1, 2, 4, 8, 12]


def main():
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists("results/mp.csv")
    with open("results/mp.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["p", "position", "n_sim", "mean_time_s", "std_s", "nps"])
        for p in PROCS:
            with Pool(p) as pool:
                mp_mcts(Board(), 200 * p, 0, pool, p)           # warm-up, not timed
                for name, moves in POSITIONS.items():
                    pos = make_position(moves)
                    times = []
                    for rep in range(REPEATS):
                        t0 = time.perf_counter()
                        mp_mcts(pos, N_SIM, rep, pool, p)
                        times.append(time.perf_counter() - t0)
                    mean, sd = statistics.mean(times), statistics.stdev(times)
                    print(f"p={p:2d} {name:6s} Tp={mean:7.3f}s +- {sd:.3f}  NPS={N_SIM/mean:9.0f}")
                    w.writerow([p, name, N_SIM, round(mean, 4), round(sd, 4), round(N_SIM / mean)])


if __name__ == "__main__":      # REQUIRED on Windows, or it starts processes endlessly
    main()