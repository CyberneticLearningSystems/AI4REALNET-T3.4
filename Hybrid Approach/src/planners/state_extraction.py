# src/planners/state_extraction.py
from __future__ import annotations
from typing import Tuple
import networkx as nx
from flatland.envs.rail_env import RailEnv

from flatland.core.grid.grid4 import Grid4Transitions

Node = Tuple[int, int, int]  # (row, col, dir)

def _turn_type(d: int, nd: int) -> str:
    if nd == d:
        return "F"
    if nd == (d - 1) % 4:
        return "L"
    if nd == (d + 1) % 4:
        return "R"
    return "U"

def build_rail_digraph(env: RailEnv, base_cost: float = 1.0) -> nx.DiGraph:
    G = nx.DiGraph()
    H, W = env.height, env.width

    for r in range(H):
        for c in range(W):
            for d in range(4):
                cell_trans = env.rail.grid[(r, c)]
                trans = env.rail.transitions.get_transitions(cell_trans, d)  # (N,E,S,W)
                if not trans or sum(int(x) for x in trans) == 0:
                    continue

                node = (r, c, d)
                if node not in G:
                    G.add_node(node, type="rail")

                for nd in range(4):
                    if int(trans[nd]) != 1:
                        continue

                    if nd == 0:   nr, nc = r - 1, c
                    elif nd == 1: nr, nc = r, c + 1
                    elif nd == 2: nr, nc = r + 1, c
                    else:         nr, nc = r, c - 1

                    if not (0 <= nr < H and 0 <= nc < W):
                        continue

                    nxt = (nr, nc, nd)
                    if nxt not in G:
                        G.add_node(nxt, type="rail")

                    G.add_edge(
                        node, nxt,
                        l=float(base_cost),
                        learned_l=float(base_cost),
                        turn=_turn_type(d, nd),
                    )
    return G