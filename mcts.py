import math
import random
from board import Board

def random_playout(board, rng):
    b= board.copy()
    moves = b.legal_moves()
    rng.shuffle(moves)

    for m in moves:
        mover = b.player
        if b.play(m):
            return mover
    return 0



class Node:
    __slots__ = ("move", "parent", "children", "untried", "visits", "wins", "mover")

    def __init__(self, move, parent, mover, untried):
        self.move = move
        self.parent = parent
        self.mover = mover
        self.untried = untried
        self.children =[]
        self.visits=0
        self.wins= 0.0

def best_child(node, c):
    log_n = math.log(node.visits)
    best, best_score = None, -1.0
    for ch in node.children:
        score = ch.wins / ch.visits + c * math.sqrt(log_n / ch.visits)  # UCT formula
        if score > best_score:
            best, best_score = ch, score
    return best

def forced_move(board):
    """Win now if possible, otherwise block the opponent's winning move."""
    cand = board.candidate_moves()
    for m in cand:                      # 1. can I win right now?
        if board.copy().play(m):
            return m
    for m in cand:                      # 2. would the opponent win there?
        b = board.copy()
        b.player = 3 - b.player
        if b.play(m):
            return m
    return None

def mcts(root_board, n_sim, seed=0, c=1.414):
    rng = random.Random(seed)
    fm = forced_move(root_board)                       # NEW
    if fm is not None:                                 # NEW
        return fm, {fm: 1}                             # NEW
    root = Node(None, None, 3 - root_board.player, root_board.candidate_moves())   # changed
    
    for _ in range(n_sim):
        node = root
        board = root_board.copy()
        winner , done = 0,False 

        while not node.untried and node.children:
            node = best_child(node, c)
            if board.play(node.move): 
                winner, done = node.mover , True
                break

        if not  done and node.untried:
            move = node.untried.pop(rng.randrange(len(node.untried)))
            mover = board.player
            won = board.play(move)
            untried = [] if (won or board.is_full()) else board.candidate_moves()
            child = Node(move, node, mover, untried)
            node.children.append(child)
            node = child
            if won :
                winner, done = mover, True

        if not done:
            winner = random_playout(board,rng)

        while node is not None:
            node.visits += 1
            if winner == 0:
                node.wins +=0.5
            elif winner == node.mover:
                node.wins +=  1.0
            node = node.parent

    visits = {ch.move: ch.visits for ch in root.children}
    best_move = max (visits, key= visits.get)
    return best_move, visits
