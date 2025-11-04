import jax
import random
import jax.numpy as jnp
from dfax import dfa2dfax, dfax2dfa, DFAx
from dfax.utils import visualize
# from dfa_samplers import RADSampler
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler

sampler = ReachAvoidSampler(max_size=5, p=None)
# sampler = ReachSampler()
# sampler = ReachAvoidSampler()
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        # [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        # [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
key = jax.random.PRNGKey(31)

dfa_1 = DFAx(
    start=0,
    transitions=jnp.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 4],
        [2, 1, 1, 1, 1, 1, 1, 1, 4, 1],
        [2, 2, 2, 2, 2, 3, 2, 2, 2, 4],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ]),
    labels=jnp.array([False, False, False, True, False])
).minimize()
dfa_2 = DFAx(
    start=0,
    transitions=jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    ]),
    labels=jnp.array([True, False, False, False, False])
).minimize()
my_dfas = [dfa_1, dfa_2]
for dfax in my_dfas:
	visualize(dfax)
input(">>")

reject_count = 0
total = 0
for _ in range(100):
	key, subkey = jax.random.split(key)
	dfax = sampler.sample(subkey)
	visualize(dfax)
	is_reject = [bool(dfax.is_reach[s]) and bool(jnp.all(dfax.transitions[s] == s)) and not bool(dfax.labels[s]) for s in range(dfax.max_n_states)]
	reject_count += any(is_reject)
	total += 1
	# dfa = dfax2dfa(dfax)
	# inputs = list(dfa.inputs)
	# w = dfa.find_word()
	# for a in w:
	# 	dfax = dfax.advance(a).minimize()
	# 	# visualize(dfax)
	# 	if dfax.n_states > 1:
	# 		is_reject = [bool(dfax.is_reach[s]) and bool(jnp.all(dfax.transitions[s] == s)) and not bool(dfax.labels[s]) for s in range(dfax.max_n_states)]
	# 		reject_count += any(is_reject)
	# 		total += 1
print(reject_count/total)
input(">>")

n = 1_000

for i in range(n):
	key, subkey = jax.random.split(key)
	dfax = sampler.sample(subkey)
	dfa = dfax2dfa(dfax)

	inputs = list(dfa.inputs)

	w = dfa.find_word()

	for a in w:
		dfa = dfa.advance([a]).minimize()
		dfax = dfax.advance(a).minimize()
		assert dfax == dfa2dfax(dfa).canonicalize()
		assert dfax.reward() == dfa2dfax(dfa).reward()
		assert dfax2dfa(dfax) == dfa

	print(f"Test completed for {i + 1} samples.", end="\r")

print(f"Test completed for {n} samples.")

