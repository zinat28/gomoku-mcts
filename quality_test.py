import csv, os, random, sys
from board import Board
from mcts import mcts
from gpu_mcts import gpu_mcts


def opening(seed, plies=4):
    rng = random.Random(seed)
    b = Board()
    for _ in range(plies):
        b.play(rng.choice(b.candidate_moves()))
    return b


def play_game(b, f_black, f_white, seed):
    b = b.copy()
    while True:
        f = f_black if b.player == 1 else f_white
        mv = f(b, seed + b.moves_played)
        mover = b.player
        if b.play(mv):
            return mover
        if b.is_full():
            return 0


def wilson(score, n, z=1.96):
    ph = score / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def main():
    n_games = int(sys.argv[1]); seq_sims = int(sys.argv[2]); gpu_sims = int(sys.argv[3])
    leaves = int(sys.argv[4]); k = int(sys.argv[5])
    f_gpu = lambda b, s: gpu_mcts(b, gpu_sims, s, leaves, k)[0]
    f_seq = lambda b, s: mcts(b, seq_sims, s)[0]
    gpu_w = seq_w = draws = 0
    for g in range(n_games):
        base = opening(g // 2)                 # each opening is played twice, colours swapped
        gpu_black = (g % 2 == 0)
        w = play_game(base, f_gpu if gpu_black else f_seq,
                      f_seq if gpu_black else f_gpu, g * 1000)
        gcol = 1 if gpu_black else 2
        if w == 0: draws += 1
        elif w == gcol: gpu_w += 1
        else: seq_w += 1
        print(f"game {g+1}/{n_games}: GPU {gpu_w}  SEQ {seq_w}  draws {draws}", flush=True)
    score = gpu_w + 0.5 * draws
    lo, hi = wilson(score, n_games)
    print(f"GPU score {score/n_games:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists("results/quality.csv")
    with open("results/quality.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["games", "seq_sims", "gpu_sims", "leaves", "k", "gpu_wins",
                        "seq_wins", "draws", "gpu_score", "ci_low", "ci_high"])
        w.writerow([n_games, seq_sims, gpu_sims, leaves, k, gpu_w, seq_w, draws,
                    round(score / n_games, 3), round(lo, 3), round(hi, 3)])


if __name__ == "__main__":
    main()