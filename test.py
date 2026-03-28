import jax
import time
import operator
import jax.numpy as jnp
from dfax import dfa2dfax, dfax2dfa
from dfax.samplers import ReachAvoidSampler
from dfax.utils import dfax2prompt, prompt2dfax


BINARY_OPS = {
	"AND": operator.and_,
	"OR": operator.or_,
	"XOR": operator.xor,
	"SUB": lambda a, b: a & (~b),
}

KEY = jax.random.PRNGKey(0)

N_SAMPLES = 100


def test_prompt_and_advance(sampler):
	key = KEY

	for i in range(N_SAMPLES):
		key, subkey = jax.random.split(key)
		dfax = sampler.sample(subkey)
		prompt = dfax2prompt(dfax)
		_dfax = prompt2dfax(prompt)
		assert dfax == _dfax, f"Original and reconstructed DFAx do not match for sample {i + 1}."
		
		dfa = dfax2dfa(dfax)
		w = dfa.find_word()
		if w is None:
			continue

		for a in w:
			dfa = dfa.advance([a]).minimize()
			dfax = dfax.advance(a).minimize()
			assert dfax == dfa2dfax(dfa).normalize()
			assert dfax.reward() == dfa2dfax(dfa).reward()
			assert dfax2dfa(dfax) == dfa

		print(f"Tests completed for {i + 1} samples.", end="\r")

	print("All tests passed for prompt and advance.")


def test_binary_ops(sampler):
	key = KEY
	
	for name, op in BINARY_OPS.items():

		for i in range(N_SAMPLES):
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
			assert dfa2dfax(prod_dfa).minimize() == prod_dfax

			print(f"Test completed for {i + 1} samples.", end="\r")

		print(f"Finished running tests for {name} implementation.")
	
	print("All tests passed for binary operations.")


def test_find_word(sampler):
	key = KEY

	for i in range(N_SAMPLES * 10):
		key, subkey = jax.random.split(key)
		dfax = sampler.sample(subkey)
		word = dfax.find_word(key)

		adv_dfax = dfax.advance(word).minimize()
		dfa_word = word[:jnp.sum(word != -1)].tolist()
		adv_dfa = dfax2dfa(dfax).advance(dfa_word).minimize()

		assert adv_dfax == dfa2dfax(adv_dfa)
		assert dfax2dfa(adv_dfax) == adv_dfa

		print(f"Test completed for {i + 1} samples.", end="\r")
	
	print("All tests passed for find word.".ljust(40))


def test_shrink(dfa_sizes):
	key = KEY

	for size in dfa_sizes:
		sampler = ReachAvoidSampler(max_size=size)
		shrink_times = []
		
		for name, op in BINARY_OPS.items():
			
			for i in range(N_SAMPLES // 5):
				key, subkey1 = jax.random.split(key)
				key, subkey2 = jax.random.split(key)

				dfax1 = sampler.sample(subkey1)
				dfax2 = sampler.sample(subkey2)

				assert dfax1.transitions.shape[0] == size
				assert dfax2.transitions.shape[0] == size

				prod_dfax = op(dfax1, dfax2).minimize()
				prod_dfa = op(dfax2dfa(dfax1).minimize(), dfax2dfa(dfax2).minimize()).minimize()

				assert prod_dfax.transitions.shape[0] == size * size

				start = time.perf_counter()
				prod_dfax_shrunk = prod_dfax.shrink()
				end = time.perf_counter()
				shrink_times.append(end - start)

				assert prod_dfax == prod_dfax_shrunk
				assert prod_dfax_shrunk.transitions.shape[0] == sum(prod_dfax_shrunk.is_reach)

				assert prod_dfax_shrunk == dfa2dfax(prod_dfa).minimize()
				assert dfax2dfa(prod_dfax_shrunk).minimize() == prod_dfa

				assert prod_dfax == dfa2dfax(prod_dfa).minimize()
				assert dfax2dfa(prod_dfax).minimize() == prod_dfa

				print(f"Test completed for {i + 1} samples (shrink on {name}).".ljust(50), end="\r")

		avg_shrink_time = sum(shrink_times) / len(shrink_times)
		print(f"Finished running tests for {size}-state DFAs "
			  f"with average shrink time of {avg_shrink_time:.4f} seconds.")
	
	print("All tests passed for shrink.")

def test_expand(sampler, expand_size):
	key = KEY

	for i in range(N_SAMPLES * 50):
		key, subkey = jax.random.split(key)
		dfax = sampler.sample(subkey)

		assert dfax.transitions.shape[0] == sampler.max_size
		
		dfax_expanded = dfax.expand(expand_size)

		assert dfax == dfax_expanded
		assert dfax_expanded.transitions.shape[0] == expand_size

		print(f"Test completed for {i + 1} samples.", end="\r")

	print("All tests passed for expand.".ljust(40))


def main():
	sampler = ReachAvoidSampler(max_size=5, p=None)
	dfa_sizes = [5, 10, 20]
	expand_size = 50

	test_prompt_and_advance(sampler)
	test_binary_ops(sampler)
	test_find_word(sampler)
	test_shrink(dfa_sizes)
	test_expand(sampler, expand_size)


if __name__ == '__main__':
	main()
