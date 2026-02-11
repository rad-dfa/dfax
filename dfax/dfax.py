import jax
import chex
import jax.numpy as jnp
from flax import struct


@jax.jit
def DFAx(start, transitions, labels):
    n_states, n_tokens = transitions.shape
    trans_flat = transitions.flatten()
    is_reach_init = jnp.zeros((n_states,), dtype=bool).at[start].set(True)

    def step(is_reach: jnp.ndarray) -> jnp.ndarray:
        reach_repeat = jnp.repeat(is_reach, n_tokens)
        dest_counts = jnp.zeros((n_states,), dtype=jnp.int32).at[trans_flat].add(reach_repeat)
        return dest_counts > 0

    def cond(pair):
        prev_is_reach, curr_is_reach = pair
        return jnp.any(prev_is_reach != curr_is_reach)

    def body(pair):
        prev_is_reach, curr_is_reach = pair
        next_is_reach = step(curr_is_reach)
        return (curr_is_reach, next_is_reach)

    is_reach, _ = jax.lax.while_loop(cond, body, (is_reach_init, step(is_reach_init)))

    return _DFAx(start=start,
                 transitions=transitions,
                 labels=labels,
                 is_reach=is_reach)


@struct.dataclass
class _DFAx:
    start: int
    transitions: jnp.ndarray
    labels: jnp.ndarray
    is_reach: jnp.ndarray

    @property
    def n_states(self):
        return jnp.sum(self.is_reach)

    @property
    def max_n_states(self):
        return self.transitions.shape[0]

    @property
    def n_tokens(self):
        return self.transitions.shape[1]

    @property
    def is_reach_tile(self) -> jnp.ndarray:
        return jnp.tile(self.is_reach.reshape(-1, 1), (1, self.n_tokens))

    @jax.jit
    def __eq__(self, other: "DFAx") -> jnp.ndarray:
        n = min(self.max_n_states, other.max_n_states)

        start_eq = (self.start == other.start)
        transitions_eq = jnp.where(
            self.n_tokens == other.n_tokens,
            jnp.all(self.transitions[:n] == other.transitions[:n]),
            False
        )
        labels_eq = jnp.all(self.labels[:n] == other.labels[:n])

        return jnp.logical_and(start_eq, jnp.logical_and(transitions_eq, labels_eq))

    def __hash__(self) -> int:
        from dfax import dfax2dfa
        dfa = dfax2dfa(self)
        return dfa.__hash__()

    @jax.jit
    def advance(self, symbol: int) -> "DFAx":
        return DFAx(
            start=jnp.where(
                jnp.logical_and(symbol >= 0, symbol < self.n_tokens),
                self.transitions[self.start, symbol],
                self.start
            ),
            transitions=self.transitions,
            labels=self.labels)

    @jax.jit
    def mutate(self, key: chex.PRNGKey) -> "DFAx":
        key, k1, k2 = jax.random.split(key, 3)

        flat_is_reach_tile = self.is_reach_tile.flatten()

        scores, indices = jax.lax.top_k(jnp.where(flat_is_reach_tile, 1, 0), flat_is_reach_tile.size)
        num_valid = jnp.sum(scores > 0)
        valid_idx = jax.random.randint(k1, (), 0, num_valid)
        flat_idx = indices[valid_idx]
        s, a = jnp.divmod(flat_idx, self.n_tokens)

        scores, indices = jax.lax.top_k(jnp.where(self.is_reach, 1, 0), self.max_n_states)
        num_valid = jnp.sum(scores > 0)
        valid_idx = jax.random.randint(k2, (), 0, num_valid)
        t = indices[valid_idx]

        transitions = self.transitions.at[s, a].set(t)

        return DFAx(start=self.start,
                    transitions=transitions,
                    labels=self.labels)

    @jax.jit
    def mutate_reject_lang(self, key: chex.PRNGKey) -> "DFAx":
        key, k1, k2 = jax.random.split(key, 3)

        is_self_loop = self.transitions == jnp.arange(self.max_n_states)[:, None]
        sa_mask = jnp.logical_and(
            jnp.logical_and(self.is_reach_tile, is_self_loop),
            jnp.logical_not(self.labels)[:, None]
        ).flatten()

        scores, indices = jax.lax.top_k(jnp.where(sa_mask, 1, 0), sa_mask.size)
        num_valid = jnp.sum(scores > 0)
        valid_idx = jax.random.randint(k1, (), 0, num_valid)
        flat_idx = indices[valid_idx]
        s, a = jnp.divmod(flat_idx, self.n_tokens)

        is_sink = jnp.all(is_self_loop, axis=-1)
        is_reject = jnp.logical_and.reduce(jnp.array([self.is_reach, is_sink, jnp.logical_not(self.labels)]))
        t_mask = jnp.where(jnp.any(is_reject), is_reject, jnp.logical_not(self.is_reach))

        scores, indices = jax.lax.top_k(jnp.where(t_mask, 1, 0), t_mask.size)
        num_valid = jnp.sum(scores > 0)
        valid_idx = jax.random.randint(k2, (), 0, num_valid)
        t = indices[valid_idx]

        transitions = self.transitions.at[s, a].set(t)

        return DFAx(start=self.start,
                    transitions=transitions,
                    labels=self.labels)

    @jax.jit
    def sink_accepts(self) -> "DFAx":
        accept_mask = jnp.tile(self.labels.reshape(-1, 1), (1, self.n_tokens))
        replace_mask = accept_mask & self.is_reach_tile

        sink_indices = jnp.tile(jnp.arange(self.max_n_states).reshape(-1, 1), (1, self.n_tokens))
        transitions = jnp.where(replace_mask, sink_indices, self.transitions)

        return DFAx(
            start=self.start,
            transitions=transitions,
            labels=self.labels
        )

    @jax.jit
    def prune(self) -> "DFAx":
        pruned_transitions = jnp.where(self.is_reach_tile, self.transitions, jnp.arange(self.max_n_states)[:, None])
        pruned_labels = jnp.where(self.is_reach, self.labels, False)

        return DFAx(start=self.start,
                    transitions=pruned_transitions,
                    labels=pruned_labels)

    @jax.jit
    def minimize(self) -> "DFAx":
        return self.naivePR().prune().canonicalize()

    @jax.jit
    def naivePR(self) -> "DFAx":
        # Algorithm 2 from https://arxiv.org/pdf/2410.22764
        q_f = jnp.argmax(jnp.where(self.is_reach, self.labels, 0))
        q_n = jnp.argmin(jnp.where(self.is_reach, self.labels, 1))
        block = jnp.where(self.labels, q_f, q_n)
        block = jnp.where(self.is_reach, block, jnp.arange(self.max_n_states))

        qs = jnp.arange(self.max_n_states)
        as_ = jnp.arange(self.n_tokens)
        qas = jnp.stack(jnp.meshgrid(qs, as_, indexing="ij"), axis=-1).reshape(-1, 2)

        def iteration(state):
            block, _ = state

            def elect(q_a):
                q, a = q_a
                return jax.lax.cond(
                    self.is_reach[q],
                    lambda _: (
                        block[q],
                        jnp.where(
                            block[self.transitions[q, a]] != block[self.transitions[block[q], a]],
                            q,
                            -1
                        )
                    ),
                    lambda _: (block[q], -1),
                    operand=None
                )
            blk_idx, leaders = jax.vmap(elect)(qas)
            new_leader = jax.ops.segment_max(leaders, blk_idx, num_segments=self.max_n_states)

            def assign(q_a):
                q, a = q_a
                return jax.lax.cond(
                    self.is_reach[q],
                    lambda _: (
                        q,
                        jnp.where(
                            (block[self.transitions[q, a]] != block[self.transitions[block[q], a]]),
                            new_leader[block[q]],
                            -1
                        )
                    ),
                    lambda _: (q, -1),
                    operand=None
                )
            qs_idx, new_vals = jax.vmap(assign)(qas)
            new_block = jax.ops.segment_max(new_vals, qs_idx, num_segments=self.max_n_states)
            new_block = jnp.where(new_block < 0, block, new_block)

            return (new_block, jnp.any(new_block != block))

        block, _ = jax.lax.while_loop(
            lambda s: s[1],
            lambda s: iteration((s[0], False)),
            (block, True)
        )

        minimized_start       = block[self.start]
        minimized_labels      = self.labels[block]
        minimized_transitions = block[self.transitions]

        return DFAx(start=minimized_start,
                    transitions=minimized_transitions,
                    labels=minimized_labels)

    @jax.jit
    def canonicalize(self) -> "DFAx":
        old_to_new = (-jnp.ones((self.max_n_states,), dtype=jnp.int32)).at[self.start].set(0)
        visited    = jnp.zeros((self.max_n_states,), dtype=bool).at[self.start].set(True)
        queue      = (-jnp.ones((self.max_n_states,), dtype=jnp.int32)).at[0].set(self.start)
        head       = 0
        tail       = (head + 1) % self.max_n_states
        count      = 0

        def cond(carry):
            _, _, _, head, tail, _ = carry
            return head != tail

        def body(carry):
            visited, old_to_new, queue, head, tail, count = carry

            current_state = queue[head]
            head = (head + 1) % self.max_n_states

            old_to_new = old_to_new.at[current_state].set(count)
            count += 1

            next_states = self.transitions[current_state]

            def push(i, carry):
                visited, queue, tail = carry
                ns = next_states[i]
                unseen = jnp.logical_not(visited[ns])

                queue = queue.at[tail].set(jnp.where(unseen, ns, queue[tail]))
                tail  = (tail + unseen) % self.max_n_states

                visited = visited.at[ns].set(True)
                return visited, queue, tail

            visited, queue, tail = jax.lax.fori_loop(
                0, self.n_tokens, push, (visited, queue, tail)
            )

            return visited, old_to_new, queue, head, tail, count

        visited, old_to_new, queue, head, tail, count = jax.lax.while_loop(
            cond, body, (visited, old_to_new, queue, head, tail, count)
        )

        mask = (old_to_new < 0)
        ranks = jnp.cumsum(mask) - 1
        fill_vals = count + ranks
        old_to_new = jnp.where(mask, fill_vals, old_to_new)

        start        = old_to_new[self.start]
        transitions  = self.transitions.at[old_to_new].set(old_to_new[self.transitions])
        labels       = self.labels.at[old_to_new].set(self.labels)

        return DFAx(start=start,
                    transitions=transitions,
                    labels=labels)

    @jax.jit
    def to_graph(self):
        srcs, tgts = jnp.meshgrid(jnp.arange(self.max_n_states), jnp.arange(self.max_n_states), indexing="ij")
        srcs = srcs.flatten()
        tgts = tgts.flatten()

        edge_index = jnp.stack([srcs, tgts])
        
        is_init = jnp.logical_and(
            self.is_reach,
            jnp.arange(self.max_n_states) == self.start
        )
        is_accept = jnp.logical_and(
            self.is_reach,
            self.labels
        )
        is_reject = jnp.logical_and(
            self.is_reach,
            jnp.all(
                jnp.logical_and(
                    self.transitions == jnp.arange(self.max_n_states)[:, None], # has all loops
                    jnp.logical_not(self.labels[:, None]) # not accepting
                ),
                axis=1
            )
        )
        is_non_terminal = jnp.logical_and(
            self.is_reach,
            jnp.logical_and(
                jnp.logical_not(is_accept),
                jnp.logical_not(is_reject)
            )
        )

        node_features = jnp.stack([is_init, is_accept, is_reject, is_non_terminal], axis=1).astype(jnp.float32)

        edge_features = jnp.logical_and(
            self.is_reach[:, None, None],
            self.transitions[:, None, :] == jnp.arange(self.max_n_states)[None, :, None] # one hot tokens
        ).astype(jnp.float32).reshape(-1, self.n_tokens)

        mask = jnp.any(edge_features != 0, axis=-1)[:, None]
        edge_features = jnp.concatenate([node_features[srcs] * mask, edge_features, node_features[tgts] * mask], axis=-1)

        graph = {
            "node_features": node_features,
            "edge_features": edge_features,
            "edge_index": edge_index,
            "current_state": jnp.array([self.start]),
            "n_states": jnp.full(self.max_n_states, self.n_states)
        }

        return graph

    @jax.jit
    def reward(self, binary: bool = False) -> float:
        is_accept = self.labels[self.start]
        start_row = self.transitions[self.start]
        start_vec = jnp.full((self.n_tokens,), self.start, dtype=start_row.dtype)
        is_sink = jnp.all(start_row == start_vec)

        return jnp.where(binary,
            jnp.where(is_accept, 1.0, 0.0),
            jnp.where(is_accept, 1.0, jnp.where(is_sink, -1.0, 0.0))
        )

