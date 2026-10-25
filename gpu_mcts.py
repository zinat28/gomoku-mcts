import random
import numpy as np
from numba import cuda
from mcts import Node, best_child, forced_move

TPB = 128   # threads per block


@cuda.jit(device=True)
def _xorshift(x):
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x


@cuda.jit(device=True)
def _check_win(cells, move, color):
    r = move // 15
    c = move % 15
    for d in range(4):
        dr = 0
        dc = 1
        if d == 1:
            dr = 1
            dc = 0
        elif d == 2:
            dr = 1
            dc = 1
        elif d == 3:
            dr = 1
            dc = -1
        count = 1
        for s in range(2):
            sign = 1 - 2 * s
            rr = r + dr * sign
            cc = c + dc * sign
            while rr >= 0 and rr < 15 and cc >= 0 and cc < 15 and cells[rr * 15 + cc] == color:
                count += 1
                rr += dr * sign
                cc += dc * sign
        if count >= 5:
            return True
    return False


@cuda.jit
def _playout_kernel(boards, players, term, k, n, seed, results):
    t = cuda.grid(1)
    if t >= n:
        return
    leaf = t // k                       # k playouts per leaf
    tw = term[leaf]
    if tw >= 0:                         # leaf is already a finished game
        results[t] = tw
        return
    cells = cuda.local.array(225, dtype=np.uint8)
    empties = cuda.local.array(225, dtype=np.int32)
    n_empty = 0
    for i in range(225):
        v = boards[leaf, i]
        cells[i] = v
        if v == 0:
            empties[n_empty] = i
            n_empty += 1
    player = players[leaf]
    x = (seed + t * 2891336453 + 1) & 0xFFFFFFFF   # per-thread seed
    if x == 0:
        x = 1
    for _ in range(4):
        x = _xorshift(x)
    winner = 0
    while n_empty > 0:                  # random playout
        x = _xorshift(x)
        j = x % n_empty
        mv = empties[j]
        empties[j] = empties[n_empty - 1]
        n_empty -= 1
        cells[mv] = player
        if _check_win(cells, mv, player):
            winner = player
            break
        player = 3 - player
    results[t] = winner


def gpu_playouts(board, n, seed):
    """n random playouts from one position. Returns counts [draw, black, white]."""
    boards = np.frombuffer(board.cells, dtype=np.uint8).reshape(1, 225).copy()
    players = np.array([board.player], dtype=np.int32)
    term = np.array([-1], dtype=np.int32)
    d_res = cuda.device_array(n, dtype=np.int32)
    blocks = (n + TPB - 1) // TPB
    _playout_kernel[blocks, TPB](cuda.to_device(boards), cuda.to_device(players),
                                 cuda.to_device(term), n, n, seed, d_res)
    return np.bincount(d_res.copy_to_host(), minlength=3)


def _select_expand(root, root_board, rng, c):
    """CPU part: selection + expansion, then add a virtual loss along the path."""
    node = root
    board = root_board.copy()
    winner = -1
    while not node.untried and node.children:
        node = best_child(node, c)
        if board.play(node.move):
            winner = node.mover
            break
    if winner < 0 and node.untried:
        move = node.untried.pop(rng.randrange(len(node.untried)))
        mover = board.player
        won = board.play(move)
        untried = [] if (won or board.is_full()) else board.candidate_moves()
        child = Node(move, node, mover, untried)
        node.children.append(child)
        node = child
        if won:
            winner = mover
    if winner < 0 and board.is_full():
        winner = 0
    n = node
    while n is not None:                # virtual loss
        n.visits += 1
        n = n.parent
    return node, board, winner


def gpu_mcts(root_board, n_sim, seed=0, n_leaves=64, k=64, c=1.414):
    """Leaf parallelization: n_leaves leaves per round, k playouts per leaf on the GPU."""
    rng = random.Random(seed)
    fm = forced_move(root_board)
    if fm is not None:
        return fm, {fm: 1}
    root = Node(None, None, 3 - root_board.player, root_board.candidate_moves())
    batch = n_leaves * k
    d_res = cuda.device_array(batch, dtype=np.int32)
    done, rnd = 0, 0
    while done < n_sim:
        threads = min(batch, n_sim - done)
        leaves = (threads + k - 1) // k
        h_boards = np.empty((leaves, 225), dtype=np.uint8)
        h_players = np.empty(leaves, dtype=np.int32)
        h_term = np.empty(leaves, dtype=np.int32)
        nodes = []
        for i in range(leaves):                          # CPU: serial tree work
            node, board, winner = _select_expand(root, root_board, rng, c)
            h_boards[i] = np.frombuffer(board.cells, dtype=np.uint8)
            h_players[i] = board.player
            h_term[i] = winner
            nodes.append(node)
        blocks = (threads + TPB - 1) // TPB              # GPU: all playouts at once
        _playout_kernel[blocks, TPB](cuda.to_device(h_boards), cuda.to_device(h_players),
                                     cuda.to_device(h_term), k, threads,
                                     seed * 1000003 + rnd * 7919, d_res)
        res = d_res[:threads].copy_to_host()
        for i, node in enumerate(nodes):                 # CPU: backpropagation
            part = res[i * k: min((i + 1) * k, threads)]
            counts = np.bincount(part, minlength=3)
            cnt = len(part)
            n_ = node
            while n_ is not None:
                n_.visits += cnt - 1                     # remove virtual loss, add real
                n_.wins += float(counts[n_.mover]) + 0.5 * float(counts[0])
                n_ = n_.parent
        done += threads
        rnd += 1
    visits = {ch.move: ch.visits for ch in root.children}
    return max(visits, key=visits.get), visits
