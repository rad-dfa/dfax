import jax
from dfa import DFA
from dfax import DFAx
import networkx as nx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def dfax2prompt(dfax: DFAx):
    _prompt = "DFA\n"
    _prompt += "STATES: " + ", ".join(f"S{i}" for i in range(dfax.n_states)) + "\n"
    _prompt += "TOKENS: " + ", ".join(f"T{i}" for i in range(dfax.n_tokens)) + "\n"
    _prompt += "INIT_STATE: " + str(dfax.start) + "\n"
    _prompt += "ACCEPTING_STATES: " + ", ".join(f"S{i}" for i in range(dfax.n_states) if dfax.labels[i]) + "\n"
    _prompt += "TRANSITIONS:\n"
    for s in range(dfax.n_states):
        for a in range(dfax.n_tokens):
            t = dfax.transitions[s, a]
            if s != t:
                _prompt += f"    S{s} -T{a}-> S{t}\n"
    _prompt += "    For every state Si and token Tj not listed above, the transition goes back to the same state:\n"
    _prompt += "        Si -Tj-> Si for all other state-token pairs.\n"
    return _prompt


def dfa2dfax(dfa: DFA) -> DFAx:
    states = dfa.states()
    inputs = dfa.inputs
    start = dfa.start
    transitions = jnp.array([[dfa._transition(s, a) for a in inputs] for s in states])
    labels = jnp.array([dfa._label(s) for s in states])
    tmp = DFAx(
        start=start,
        transitions=transitions,
        labels=labels
    )
    return tmp


def dfax2dfa(dfax: DFAx) -> DFA:
    inputs = set(range(dfax.transitions.shape[1]))

    def transition(s, a):
        return int(dfax.transitions[s, a])

    def label(s):
        return bool(dfax.labels[s])

    return DFA(
        start=int(dfax.start),
        inputs=inputs,
        transition=transition,
        label=label,
    )


@jax.jit
def batch2graph(batch):
    if batch["node_features"].ndim == 2 and batch["edge_features"].ndim == 2 and batch["edge_index"].ndim == 2 and batch["current_state"].ndim == 1:
        return batch

    batch_size, n_nodes, _ = batch["node_features"].shape
    node_features = jnp.concatenate(batch["node_features"])
    edge_features = jnp.concatenate(batch["edge_features"])
    offset = (jnp.arange(batch_size, dtype=jnp.int32) * n_nodes)
    edge_index = jnp.concatenate(batch["edge_index"] + offset[:, None, None], axis=1)
    current_state = (batch["current_state"].reshape(batch_size, -1) + offset[:, None]).flatten()
    n_states = jnp.concatenate(batch["n_states"])

    graph = {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_index": edge_index,
        "current_state": current_state,
        "n_states": n_states
    }

    return graph

@jax.jit
def list2batch(graphs):
        node_features_batch = jnp.stack([graph["node_features"] for graph in graphs], axis=0)
        node_features_batch = node_features_batch if node_features_batch.ndim == 3 else jnp.concatenate(node_features_batch, axis=0)

        edge_features_batch = jnp.stack([graph["edge_features"] for graph in graphs], axis=0)
        edge_features_batch = edge_features_batch if edge_features_batch.ndim == 3 else jnp.concatenate(edge_features_batch, axis=0)

        edge_index_batch = jnp.stack([graph["edge_index"] for graph in graphs], axis=0)
        edge_index_batch = edge_index_batch if edge_index_batch.ndim == 3 else jnp.concatenate(edge_index_batch, axis=0)

        current_state_batch = jnp.stack([graph["current_state"] for graph in graphs], axis=0)
        current_state_batch = current_state_batch if current_state_batch.ndim == 2 else jnp.concatenate(current_state_batch, axis=0)

        n_states_batch = jnp.stack(jnp.array([graph["n_states"] for graph in graphs]), axis=0)
        n_states_batch = n_states_batch if n_states_batch.ndim == 2 else jnp.concatenate(n_states_batch, axis=0)

        batch = {
            "node_features": node_features_batch,
            "edge_features": edge_features_batch,
            "edge_index": edge_index_batch,
            "current_state": current_state_batch,
            "n_states": n_states_batch,
        }

        return batch


def visualize(dfax, label_states=False, save_path=None):
    n_states, n_tokens = dfax.transitions.shape

    G = nx.DiGraph()
    for s in range(n_states):
        if dfax.is_reach[s]:
            G.add_node(s, label=str(s))

    edges = {}
    for s in range(n_states):
        s = int(s)
        for a in range(n_tokens):
            a = int(a)
            t = int(dfax.transitions[s, a])
            if s != t:
                if (s, t) not in edges:
                    edges[(s, t)] = [str(a)]
                else:
                    edges[(s, t)].append(str(a))

    for (s, t) in edges:
        G.add_edge(s, t, label=edges[(s, t)])

    # dummy start node
    dummy_start = ""
    G.add_node(dummy_start)
    G.add_edge(dummy_start, int(dfax.start))

    pos = nx.shell_layout(G)
    # pos = nx.planar_layout(G)
    start_pos = pos[int(dfax.start)]
    pos[dummy_start] = (start_pos[0] - 0.5, start_pos[1])

    accept_nodes = [s for s in G.nodes() if s != dummy_start and dfax.labels[s]]
    reject_nodes = [s for s in G.nodes() if s != dummy_start and not dfax.labels[s] and jnp.all(dfax.transitions[s] == s)]
    undecd_nodes = [s for s in G.nodes() if s != dummy_start and not dfax.labels[s]]

    # draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=undecd_nodes, node_size=1200,
                           node_color="white", edgecolors="black", linewidths=2)
    nx.draw_networkx_nodes(G, pos, nodelist=accept_nodes, node_size=1200,
                           node_color="#88E788", edgecolors="black", linewidths=2)
    nx.draw_networkx_nodes(G, pos, nodelist=reject_nodes, node_size=1200,
                           node_color="#FF746C", edgecolors="black", linewidths=2)

    if label_states:
        nx.draw_networkx_labels(G, pos, font_size=20, font_weight="bold")

    ax = plt.gca()

    # draw edges
    for (u, v) in G.edges():
        if u == dummy_start:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrows=True, arrowsize=20, node_size=1200)
            continue

        if G.has_edge(v, u):
            rad = 0.25
        else:
            rad = 0.0

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrows=True, arrowsize=20,
                               connectionstyle=f"arc3,rad={rad}", node_size=1200)

        # --- draw tokens along the curved edge ---
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        n = len(edges[(u, v)])
        for i, a in enumerate(edges[(u, v)]):
            ratio = (i + 1) / (n + 1)

            # Compute midpoint along arc3 curve
            if rad != 0:
                # Control point for the quadratic Bezier
                xm_ctrl = (x0 + x1) / 2 + rad * (y1 - y0)
                ym_ctrl = (y0 + y1) / 2 - rad * (x1 - x0)

                # Quadratic Bezier formula
                xm = (1 - ratio) ** 2 * x0 + 2 * (1 - ratio) * ratio * xm_ctrl + ratio ** 2 * x1
                ym = (1 - ratio) ** 2 * y0 + 2 * (1 - ratio) * ratio * ym_ctrl + ratio ** 2 * y1
            else:
                xm = x0 * (1 - ratio) + x1 * ratio
                ym = y0 * (1 - ratio) + y1 * ratio

            circle = patches.Circle((xm, ym), 0.08, facecolor="gold", edgecolor="orange", lw=1.5, zorder=5)
            ax.add_patch(circle)
            ax.text(xm, ym, a, ha="center", va="center", fontsize=16, color="black", weight="bold", zorder=6)

    plt.axis("equal")
    plt.tight_layout()
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close()

