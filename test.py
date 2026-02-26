import jax
import time
import random
import operator
import jax.numpy as jnp
from dfax import dfa2dfax, dfax2dfa, DFAx
from dfax.utils import visualize, dfax2prompt, prompt2dfax, data2sampler
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler, DataSampler

BINARY_OPS = {
	"AND": operator.and_,
	"OR": operator.or_,
	"XOR": operator.xor,
	"SUB": lambda a, b: a & (~b),
}

key = jax.random.PRNGKey(0)

sampler = ReachAvoidSampler(max_size=5, p=None)
# sampler = ReachSampler()
# sampler = ReachAvoidSampler()
# sampler = data2sampler("dataset.pkl")

n = 100

dfa_sizes = [10, 20, 50, 100]

for size in dfa_sizes:

	sampler = RADSampler(max_size=size)
	shrink_times = []
	for i in range(n):
		key, subkey1 = jax.random.split(key)
		key, subkey2 = jax.random.split(key)

		dfax1 = sampler.sample(subkey1)
		dfax2 = sampler.sample(subkey2)

		assert dfax1.transitions.shape[0] == size
		assert dfax2.transitions.shape[0] == size

		dfax_and = dfax1 & dfax2

		assert dfax_and.transitions.shape[0] == size * size

		start = time.perf_counter()
		dfax_and = dfax_and.shrink()
		end = time.perf_counter()

		shrink_times.append(end - start)

		assert dfax_and.transitions.shape[0] == sum(dfax_and.is_reach)

		print(f"Test completed for {i + 1} samples.", end="\r")

	avg_shrink_time = sum(shrink_times) / len(shrink_times)
	print(f"Finished running shrink tests for {sampler.max_size}-state DFAs "
	   	  f"with average shrink time of {avg_shrink_time:.4f} seconds.")

sampler = ReachAvoidSampler(max_size=5, p=None)

for name, op in BINARY_OPS.items():

	for i in range(n):
		key, subkey1 = jax.random.split(key)
		key, subkey2 = jax.random.split(key)

		dfax1 = sampler.sample(subkey1)
		dfax2 = sampler.sample(subkey2)
		prod_dfax = op(dfax1, dfax2).minimize()

		dfa1 = dfax2dfa(dfax1)
		dfa2 = dfax2dfa(dfax2)
		prod_dfa = op(dfa1, dfa2).minimize()

		assert dfa2dfax(dfa1) == dfax1
		assert dfa2dfax(dfa2) == dfax2
		assert dfax2dfa(prod_dfax).minimize() == prod_dfa
		assert dfa2dfax(prod_dfa).canonicalize() == prod_dfax

		print(f"Test completed for {i + 1} samples.", end="\r")

	print(f"Finished running tests for {name} implementation.")


for i in range(n):
	key, subkey = jax.random.split(key)
	dfax = sampler.sample(subkey)
	prompt = dfax2prompt(dfax)
	_dfax = prompt2dfax(prompt)
	assert dfax == _dfax, f"Original and reconstructed DFAx do not match for sample {i + 1}."
	dfa = dfax2dfa(dfax)

	inputs = list(dfa.inputs)

	w = dfa.find_word()

	if w is None:
		continue

	for a in w:
		dfa = dfa.advance([a]).minimize()
		dfax = dfax.advance(a).minimize()
		assert dfax == dfa2dfax(dfa).canonicalize()
		assert dfax.reward() == dfa2dfax(dfa).reward()
		assert dfax2dfa(dfax) == dfa

	print(f"Test completed for {i + 1} samples.", end="\r")

print(f"Test completed for {n} samples.")

