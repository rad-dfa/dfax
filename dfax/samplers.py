import jax
import chex
from dfax import DFAx
import jax.numpy as jnp
from flax import struct
from functools import partial


# Base sampler: holds parameters
@struct.dataclass
class DFASampler:
    n_tokens: int = 10
    min_size: int = 3
    max_size: int = 10
    p: float | None = None

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key: chex.PRNGKey, accept_set: jnp.array=None, reject_set: jnp.array=None) -> DFAx:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def sample_n(self, key: chex.PRNGKey):
        if self.p is not None:
            values = jnp.arange(self.min_size, self.max_size + 1)
            weights = self.p ** values
            weights = weights / jnp.sum(weights)
            idx = jax.random.choice(key, values, p=weights)
            return idx
        else:
            return jax.random.randint(key, (), self.min_size, self.max_size + 1)

    @partial(jax.jit, static_argnums=(0,))
    def trivial(self, label):
        start = 0
        transitions = jnp.tile(jnp.arange(self.max_size).reshape(-1, 1), (1, self.n_tokens))
        labels = jnp.zeros((self.max_size,), dtype=bool).at[start].set(label)
        return DFAx.create(
            start=start,
            transitions=transitions,
            labels=labels
        )


# Precomputed data sampler -- initialize using data2sampler utility function
@struct.dataclass
class DataSampler(DFASampler):
    dfax_array: DFAx = None
    embd_array: jnp.ndarray = None

    @jax.jit
    def sample(self, key: chex.PRNGKey, accept_set: jnp.array=None, reject_set: jnp.array=None) -> DFAx:

        if accept_set is None and reject_set is None:
            idx = jax.random.randint(key, (), 0, self.dfax_array.labels.shape[0])
        else:
            def accepts(dfax: DFAx, w: jnp.ndarray):
                final = dfax.advance(w)
                return final.labels[final.start]

            def accepts_all(dfax: DFAx, strings: jnp.ndarray):
                return jnp.all(jax.vmap(lambda w: accepts(dfax, w))(strings))

            def rejects_all(dfax: DFAx, strings: jnp.ndarray):
                return jnp.all(jax.vmap(lambda w: ~accepts(dfax, w))(strings))

            accept_mask = (
                jax.vmap(lambda d: accepts_all(d, accept_set))(self.dfax_array)
                if accept_set is not None
                else jnp.ones(self.dfax_array.labels.shape[0], dtype=bool)
            )

            reject_mask = (
                jax.vmap(lambda d: rejects_all(d, reject_set))(self.dfax_array)
                if reject_set is not None
                else jnp.ones(self.dfax_array.labels.shape[0], dtype=bool)
            )

            valid_mask = accept_mask & reject_mask

            def sample_valid(_):
                probs = valid_mask / jnp.sum(valid_mask)
                return jax.random.choice(key, self.dfax_array.labels.shape[0], p=probs)

            def return_trivial(_):
                rejecting_dfax_mask = jax.vmap(lambda dfax: jnp.logical_and(jnp.logical_not(dfax.labels[dfax.start]), dfax.n_states == 1))(self.dfax_array)
                probs = rejecting_dfax_mask / jnp.sum(rejecting_dfax_mask)
                return jax.random.choice(key, self.dfax_array.labels.shape[0], p=probs)

            idx = jax.lax.cond(
                jnp.any(valid_mask),
                sample_valid,
                return_trivial,
                operand=None
            )

        return jax.tree.map(lambda x: x[idx], self.dfax_array)

    @jax.jit
    def embed(self, dfax: DFAx) -> jnp.ndarray:
        start_match = self.dfax_array.start == dfax.start
        transitions_match = jnp.all(
            self.dfax_array.transitions == dfax.transitions,
            axis=(1, 2)
        )
        labels_match = jnp.all(
            self.dfax_array.labels == dfax.labels,
            axis=1
        )

        full_match = start_match & transitions_match & labels_match
        idx = jnp.argmax(full_match)

        return self.embd_array[idx]

    @jax.jit
    def trivial(self, label):
        dfax = jax.tree.map(lambda x: x[0], self.dfax_array)
        start = 0
        transitions = jnp.tile(jnp.arange(dfax.max_n_states).reshape(-1, 1), (1, dfax.n_tokens))
        labels = jnp.zeros((dfax.max_n_states,), dtype=bool).at[start].set(label)
        return DFAx.create(
            start=start,
            transitions=transitions,
            labels=labels
        )


# Reach sampler
@struct.dataclass
class ReachSampler(DFASampler):
    prob_stutter: float = 0.9
    min_size: int = 2

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key: chex.PRNGKey, accept_set: jnp.array=None, reject_set: jnp.array=None) -> DFAx:
        if accept_set is not None or reject_set is not None:
            raise NotImplementedError("Conditioned sampling not implemented for ReachSampler.")
        key, subkey = jax.random.split(key)
        n = self.sample_n(subkey)
        success = n-1
        transitions = jnp.tile(jnp.arange(self.max_size).reshape(-1, 1), (1, self.n_tokens))
        labels = jnp.zeros(self.max_size, dtype=bool)
        labels = labels.at[success].set(True)
        transitions = transitions.at[success, :].set(success)

        def body_fn(i, carry):
            transitions, labels, key = carry
            key, k1, k2, k3 = jax.random.split(key, 4)
            perm = jax.random.permutation(k1, jnp.arange(self.n_tokens))
            row = jnp.full(self.n_tokens, i, dtype=jnp.int32)
            row = row.at[perm[0]].set(i+1)
            rest = perm[1:]
            r = jax.random.uniform(k2, (self.n_tokens-1,))
            choice = jax.random.bernoulli(k3, 0.5, (self.n_tokens-1,))
            dest = jnp.where(r <= self.prob_stutter, i, i+1)
            row = row.at[rest].set(dest)
            transitions = transitions.at[i].set(row)
            return (transitions, labels, key)

        transitions, labels, _ = jax.lax.cond(
            n == 0,
            lambda _: (jnp.tile(jnp.arange(self.max_size).reshape(-1, 1), (1, self.n_tokens)), jnp.zeros(self.max_size, dtype=bool).at[0].set(True), key),
            lambda _: jax.lax.fori_loop(0, n-1, body_fn, (transitions, labels, key)),
            operand=None
        )
        return DFAx.create(
            start=0,
            transitions=transitions,
            labels=labels
        ).minimize()


# Reach-Avoid sampler
@struct.dataclass
class ReachAvoidSampler(DFASampler):
    prob_stutter: float = 0.9

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key: chex.PRNGKey, accept_set: jnp.array=None, reject_set: jnp.array=None) -> DFAx:
        if accept_set is not None or reject_set is not None:
            raise NotImplementedError("Conditioned sampling not implemented for ReachAvoidSampler.")
        key, subkey = jax.random.split(key)
        n = self.sample_n(subkey)
        success, fail = n-2, n-1
        transitions = jnp.tile(jnp.arange(self.max_size).reshape(-1, 1), (1, self.n_tokens))
        labels = jnp.zeros(self.max_size, dtype=bool)
        labels = labels.at[success].set(True)
        transitions = transitions.at[success, :].set(success)
        transitions = transitions.at[fail, :].set(fail)

        def body_fn(i, carry):
            transitions, labels, key = carry
            key, k1, k2, k3 = jax.random.split(key, 4)
            perm = jax.random.permutation(k1, jnp.arange(self.n_tokens))
            row = jnp.full(self.n_tokens, i, dtype=jnp.int32)
            row = row.at[perm[0]].set(i+1)
            row = row.at[perm[1]].set(fail)
            rest = perm[2:]
            r = jax.random.uniform(k2, (self.n_tokens-2,))
            choice = jax.random.bernoulli(k3, 0.5, (self.n_tokens-2,))
            dest = jnp.where(r <= self.prob_stutter, i, jnp.where(choice, i+1, fail))
            row = row.at[rest].set(dest)
            transitions = transitions.at[i].set(row)
            return (transitions, labels, key)

        transitions, labels, _ = jax.lax.cond(
            n == 0,
            lambda _: (jnp.tile(jnp.arange(self.max_size).reshape(-1, 1), (1, self.n_tokens)), jnp.zeros(self.max_size, dtype=bool).at[0].set(True), key),
            lambda _: jax.lax.fori_loop(0, n-2, body_fn, (transitions, labels, key)),
            operand=None
        )
        return DFAx.create(
            start=0,
            transitions=transitions,
            labels=labels
        ).minimize()


# Reach-Avoid with random mutations
@struct.dataclass
class RADSampler(DFASampler):
    p: float | None = 0.5
    prob_stutter: float = 0.9
    max_mutations: int = 5

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key: chex.PRNGKey, accept_set: jnp.array=None, reject_set: jnp.array=None) -> DFAx:
        if accept_set is not None or reject_set is not None:
            raise NotImplementedError("Conditioned sampling not implemented for RADSampler.")
        key, subkey = jax.random.split(key)
        n = self.sample_n(subkey)
        success, fail = n-2, n-1
        transitions = jnp.tile(jnp.arange(self.max_size).reshape(-1, 1), (1, self.n_tokens))
        labels = jnp.zeros(self.max_size, dtype=bool)
        labels = labels.at[success].set(True)
        transitions = transitions.at[success, :].set(success)
        transitions = transitions.at[fail, :].set(fail)

        def body_fn(i, carry):
            transitions, labels, key = carry
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            perm = jax.random.permutation(k1, jnp.arange(self.n_tokens))
            row = jnp.full(self.n_tokens, i, dtype=jnp.int32)
            row = row.at[perm[0]].set(i+1)
            is_avoid_problem = jax.random.bernoulli(k2, 0.5)
            row = row.at[perm[1]].set(jnp.where(is_avoid_problem, fail, i))
            rest = perm[2:]
            r = jax.random.uniform(k3, (self.n_tokens-2,))
            choice = jax.random.bernoulli(k4, 0.5, (self.n_tokens-2,))
            dest = jnp.where(r <= self.prob_stutter, i, jnp.where(choice, i+1, fail))
            row = row.at[rest].set(dest)
            transitions = transitions.at[i].set(row)
            return (transitions, labels, key)

        transitions, labels, _ = jax.lax.cond(
            n == 0,
            lambda _: (jnp.tile(jnp.arange(self.max_size).reshape(-1, 1), (1, self.n_tokens)), jnp.zeros(self.max_size, dtype=bool).at[0].set(True), key),
            lambda _: jax.lax.fori_loop(0, n-2, body_fn, (transitions, labels, key)),
            operand=None
        )
        candidate = DFAx.create(
            start=0,
            transitions=transitions,
            labels=labels
        ).minimize()

        key, subkey = jax.random.split(key)
        n_mutations = jax.random.choice(subkey, self.max_mutations + 1)

        def derive(i, carry):
            k, cand = carry
            k, sk = jax.random.split(k)
            new_cand = cand.mutate(sk).sink_accepts().minimize()
            cand = jax.lax.cond(
                new_cand.n_states <= 1,
                lambda _: cand,
                lambda _: new_cand,
                operand=None
            )
            return k, cand

        _, candidate = jax.lax.fori_loop(0, n_mutations, derive, (key, candidate))

        return candidate

