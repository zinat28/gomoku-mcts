import time, statistics, csv, os, platform
from board import Board, SIZE
from mcts import mcts

N_SIM = 10000
REPEATS = 5

def idx(r, c): return r * SIZE + c

def make_position(moves):
    b = Board()
    for m in moves:
        b.play(idx(*m))
    return b

POSITIONS = {
    "empty": [],
    "early": [(7,7),(7,8),(8,8),(6,6)],
    "mid":   [(7,7),(7,8),(8,8),(6,6),(8,7),(6,7),(9,9),(10,10),(8,6),(9,7)],
    "late":  [(7,7),(7,8),(7,9),(8,7),(8,6),(6,7),(8,8),(8,9),(9,7),(9,8),
              (6,8),(7,6),(9,10),(5,8),(5,6),(9,6),(10,8),(8,10),(6,5),(10,7)],
}

def main():
    mcts(Board(), 200, seed=0)   # warm-up, not timed
    os.makedirs("results", exist_ok=True)
    print(f"CPU: {platform.processor()} | Python {platform.python_version()}")
    rows = []
    for name, moves in POSITIONS.items():
        pos = make_position(moves)
        times = []
        for rep in range(REPEATS):
            t0 = time.perf_counter()
            mcts(pos, N_SIM, seed=rep)
            times.append(time.perf_counter() - t0)
        mean, sd = statistics.mean(times), statistics.stdev(times)
        nps = N_SIM / mean
        print(f"{name:6s}  T1={mean:7.3f}s ± {sd:.3f}  NPS={nps:9.0f}")
        rows.append([name, N_SIM, round(mean, 4), round(sd, 4), round(nps)])
    with open("results/sequential.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "n_sim", "mean_time_s", "std_s", "nps"])
        w.writerows(rows)

if __name__ == "__main__":
    main()