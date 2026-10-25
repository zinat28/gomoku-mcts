import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(p):
    with open(p) as f:
        return list(csv.DictReader(f))


lap = {r["position"]: float(r["nps"]) for r in read("results/sequential.csv")}
col = {r["position"]: float(r["nps"]) for r in read("results/sequential_colab.csv")}
gpu = read("results/gpu.csv")

# ---- GPU table: Sp vs Colab T1 (fair) and vs laptop T1 ----
rows = [["gpu", "batch", "n_sim", "position", "Tp_s", "std_s", "NPS",
         "Sp_vs_colab", "Sp_vs_laptop", "Lp", "Latency_ms", "Ep_Fp_ep"]]
for r in gpu:
    tp, nps, pos = float(r["mean_time_s"]), float(r["nps"]), r["position"]
    rows.append([r["gpu"], r["batch"], r["n_sim"], pos, tp, r["std_s"], int(nps),
                 round(nps / col[pos], 1), round(nps / lap[pos], 1),
                 round(1 / tp, 2), round(tp * 1000, 1), "n/a"])
with open("results/gpu_summary.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)

# ---- Figure 1: GPU NPS vs batch size ----
by = defaultdict(list)
for r in gpu:
    by[(int(r["n_sim"]), int(r["batch"]))].append(float(r["nps"]))
plt.figure(figsize=(6, 4))
for n in sorted({k[0] for k in by}):
    bs = sorted(b for (m, b) in by if m == n)
    plt.plot(bs, [sum(by[(n, b)]) / len(by[(n, b)]) for b in bs], marker="o", label=f"N_SIM={n}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Batch size B (threads per launch)"); plt.ylabel("NPS (avg over positions)")
plt.title("GPU throughput vs batch size"); plt.legend(); plt.grid(True, which="both")
plt.tight_layout(); plt.savefig("results/fig_gpu_batch.png", dpi=200); plt.close()

# ---- CPU metrics from master.csv ----
m = defaultdict(lambda: defaultdict(list))
for r in read("results/master.csv"):
    if r["method"] == "sequential":
        continue
    key = (r["method"], int(r["p"]))
    for c in ("Sp", "Ep_percent", "Fp", "Lp", "NPS"):
        m[key][c].append(float(r[c]))
avg = lambda key, c: sum(m[key][c]) / len(m[key][c])
ps = sorted({p for _, p in m})
methods = sorted({k for k, _ in m})

# ---- Figure 2: Sp, Ep, Fp and Lp vs p ----
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for name in methods:
    ax[0].plot(ps, [avg((name, p), "Sp") for p in ps], marker="o", label=name)
    ax[1].plot(ps, [avg((name, p), "Ep_percent") for p in ps], marker="o", label=name)
    ax[2].plot(ps, [avg((name, p), "Fp") for p in ps], marker="o", label=name)
    ax[3].plot(ps, [avg((name, p), "Lp") for p in ps], marker="o", label=name)
ax[0].plot(ps, ps, "k--", label="ideal")
ax[1].axhline(100, color="k", ls="--")
for a, t in zip(ax, ("Speedup Sp", "Efficiency Ep (%)", "Effectiveness Fp", "Temporal perf. Lp")):
    a.set_title(t); a.set_xlabel("workers p"); a.grid(True)
ax[0].legend()
plt.tight_layout(); plt.savefig("results/fig_cpu_metrics.png", dpi=200); plt.close()

# ---- Figure 3: NPS of all methods ----
gpu_best = [float(r["nps"]) for r in gpu if r["batch"] == "65536" and r["n_sim"] == "10000"]
bars = [("Seq (laptop)", sum(lap.values()) / 4),
        ("MPI p=8", avg(("MPI", 8), "NPS")), ("MPI p=12", avg(("MPI", 12), "NPS")),
        ("MP p=8", avg(("multiprocessing", 8), "NPS")),
        ("Seq (Colab)", sum(col.values()) / 4),
        ("GPU T4 B=65536", sum(gpu_best) / len(gpu_best))]
plt.figure(figsize=(7, 4))
plt.bar([b[0] for b in bars], [b[1] for b in bars])
plt.yscale("log"); plt.ylabel("NPS (log scale)"); plt.xticks(rotation=25, ha="right")
plt.title("Throughput of all implementations"); plt.tight_layout()
plt.savefig("results/fig_nps_all.png", dpi=200)
print("Saved gpu_summary.csv and 3 figures in results/")