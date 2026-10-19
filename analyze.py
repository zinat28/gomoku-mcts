import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


seq = {r["position"]: float(r["mean_time_s"]) for r in read("results/sequential.csv")}
mpi = {}
for r in read("results/mpi.csv"):
    mpi[(int(r["p"]), r["position"])] = float(r["mean_time_s"])   # last run wins

ps = sorted({p for p, _ in mpi})
rows = [["position", "p", "Tp_s", "Sp", "Ep_percent", "KarpFlatt"]]

print(f"{'position':8s} {'p':>3s} {'Tp(s)':>8s} {'Sp':>6s} {'Ep(%)':>7s} {'KF':>7s}")
for pos, t1 in seq.items():
    for p in ps:
        tp = mpi[(p, pos)]
        sp = t1 / tp
        ep = sp / p * 100
        kf = (1 / sp - 1 / p) / (1 - 1 / p) if p > 1 else float("nan")
        print(f"{pos:8s} {p:3d} {tp:8.3f} {sp:6.2f} {ep:7.1f} {kf:7.3f}")
        rows.append([pos, p, round(tp, 4), round(sp, 3), round(ep, 1), round(kf, 4)])

with open("results/summary.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)

# chart: speedup vs processes, with the ideal line
plt.figure(figsize=(6, 4))
for pos, t1 in seq.items():
    plt.plot(ps, [t1 / mpi[(p, pos)] for p in ps], marker="o", label=pos)
plt.plot(ps, ps, "k--", label="ideal")
plt.xlabel("MPI processes (p)")
plt.ylabel("Speedup Sp = T1 / Tp")
plt.title("Root-parallel MCTS speedup")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/speedup.png", dpi=150)
print("Saved results/summary.csv and results/speedup.png")