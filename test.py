import jax
import random
import jax.numpy as jnp
from dfax import dfa2dfax, dfax2dfa, DFAx
from dfax.utils import visualize, dfax2prompt, prompt2dfax
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler


key = jax.random.PRNGKey(0)

sampler = ReachAvoidSampler(max_size=5, p=None)
# sampler = ReachSampler()
# sampler = ReachAvoidSampler()

n = 100

for i in range(n):
	key, subkey = jax.random.split(key)
	dfax = sampler.sample(subkey)
	prompt = dfax2prompt(dfax)
	_dfax = prompt2dfax(prompt)
	assert dfax == _dfax, f"Original and reconstructed DFAx do not match for sample {i + 1}."
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

