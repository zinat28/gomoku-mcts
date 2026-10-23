import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load(path):
    d = {}
    for r in read(path):                      # if a run was repeated, the last one wins
        d[(int(r["p"]), r["position"])] = (float(r["mean_time_s"]),
                                           float(r["std_s"]), int(r["n_sim"]))
    return d


seq = {r["position"]: (float(r["mean_time_s"]), float(r["std_s"]), int(r["n_sim"]))
       for r in read("results/sequential.csv")}
methods = {}
for name, path in (("MPI", "results/mpi.csv"), ("multiprocessing", "results/mp.csv")):
    if os.path.exists(path):
        methods[name] = load(path)

header = ["method", "position", "p", "Tp_s", "std_s", "NPS", "Sp", "Ep_percent",
          "Fp", "Lp", "KarpFlatt_ep", "Amdahl_bound", "Latency_ms"]
rows = [header]


def add(method, pos, p, tp, sd, n, t1):
    sp = t1 / tp
    ep = sp / p * 100
    fp = sp / (p * tp)            # effectiveness
    lp = 1 / tp                   # temporal performance
    if p > 1:
        kf = (1 / sp - 1 / p) / (1 - 1 / p)
        kf_s = round(kf, 4)
        bound = round(1 / kf, 1) if kf > 0 else "NA"
    else:
        kf_s, bound = "NA", "NA"
    rows.append([method, pos, p, round(tp, 4), round(sd, 4), round(n / tp),
                 round(sp, 3), round(ep, 1), round(fp, 4), round(lp, 4),
                 kf_s, bound, round(tp * 1000, 1)])


for pos, (t1, sd, n) in seq.items():
    add("sequential", pos, 1, t1, sd, n, t1)
    for name, data in methods.items():
        for p in sorted({p for p, _ in data}):
            tp, sd2, n2 = data[(p, pos)]
            add(name, pos, p, tp, sd2, n2, t1)

with open("results/master.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
for r in rows:
    print(" | ".join(str(x) for x in r))

plt.figure(figsize=(6, 4))
first = True
for name, data in methods.items():
    ps = sorted({p for p, _ in data})
    avg = [sum(seq[pos][0] / data[(p, pos)][0] for pos in seq) / len(seq) for p in ps]
    plt.plot(ps, avg, marker="o", label=name)
    if first:
        plt.plot(ps, ps, "k--", label="ideal")
        first = False
plt.xlabel("workers (p)")
plt.ylabel("average speedup Sp")
plt.title("MPI vs multiprocessing")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/speedup.png", dpi=150)
print("Saved results/master.csv and results/speedup.png")