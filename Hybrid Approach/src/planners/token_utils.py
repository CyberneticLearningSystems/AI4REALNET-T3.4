import networkx as nx

def apply_tokens_to_graph(G: nx.DiGraph, tokens):
    for token in tokens:

        if token["kind"] == "AVOID_EDGE":
            u = tuple(token["payload"]["u"])
            v = tuple(token["payload"]["v"])

            if G.has_edge(u, v):
                G[u][v]["l"] = 1000
                G[u][v]["learned_l"] = 1000