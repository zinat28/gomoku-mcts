from mcts import mcts


def _worker(args):
    board, n, seed = args
    _, visits = mcts(board, n, seed=seed)
    return visits


def mp_mcts(board, n_sim, base_seed, pool, p):
    """Root parallelization with a process pool."""
    jobs = []
    for k in range(p):
        local = n_sim // p + (1 if k < n_sim % p else 0)
        if local > 0:
            jobs.append((board, local, base_seed + k))   # distinct seed per worker

    results = pool.map(_worker, jobs)                    # run in parallel

    total = {}
    for visits in results:                               # REDUCE: sum root-child visits
        for move, n in visits.items():
            total[move] = total.get(move, 0) + n
    return max(total, key=total.get)