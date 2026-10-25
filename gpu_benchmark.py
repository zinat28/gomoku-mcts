import csv, os, statistics, time
from numba import cuda
from gpu_mcts import gpu_mcts
from benchmark import POSITIONS, make_position

CONFIGS = [(16, 16), (64, 16), (64, 64), (128, 128), (256, 256)]   # (leaves, playouts per leaf)
N_SIMS = [10000, 100000]
REPEATS = 5


def main():
    name = cuda.get_current_device().name
    name = name.decode() if isinstance(name, bytes) else name
    print("GPU:", name)
    gpu_mcts(make_position(POSITIONS["mid"]), 5000, 0, 16, 16)       # warm-up, not timed
    os.makedirs("results", exist_ok=True)
    with open("results/gpu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "leaves", "k", "batch", "n_sim", "position",
                    "mean_time_s", "std_s", "nps"])
        for n_sim in N_SIMS:
            for leaves, k in CONFIGS:
                for pname, moves in POSITIONS.items():
                    pos = make_position(moves)
                    times = []
                    for rep in range(REPEATS):
                        t0 = time.perf_counter()
                        gpu_mcts(pos, n_sim, rep, leaves, k)
                        times.append(time.perf_counter() - t0)
                    mean, sd = statistics.mean(times), statistics.stdev(times)
                    print(f"N={n_sim:6d} B={leaves*k:6d} {pname:6s} Tp={mean:7.3f}s +- {sd:.3f}  NPS={n_sim/mean:9.0f}")
                    w.writerow([name, leaves, k, leaves * k, n_sim, pname,
                                round(mean, 4), round(sd, 4), round(n_sim / mean)])


if __name__ == "__main__":
    main()
