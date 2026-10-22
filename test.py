import jax
import time
import operator
import jax.numpy as jnp
from dfax import dfa2dfax, dfax2dfa
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler, MixtureSampler
from dfax.utils import dfax2prompt, prompt2dfax, data2sampler


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


def check_sample(sampler, dfax):
	# Samples are canonical minimal DFAs padded to the sampler's size, and trivial DFAs have the same shapes
	assert dfax.transitions.shape == (sampler.max_size, sampler.n_tokens)
	assert dfax.labels.shape == (sampler.max_size,)
	assert int(dfax.start) == 0
	assert dfax == dfax.minimize()

	for label, reward in [(True, 1), (False, -1)]:
		trivial = sampler.trivial(label)
		assert all(a.shape == b.shape for a, b in zip(jax.tree.leaves(trivial), jax.tree.leaves(dfax)))
		assert trivial.reward() == reward


def state_masks(dfax):
	is_sink = jnp.all(dfax.transitions == jnp.arange(dfax.max_n_states)[:, None], axis=1) & dfax.is_reach
	is_accept = dfax.labels & dfax.is_reach
	is_reject_sink = is_sink & ~dfax.labels
	return is_accept, is_reject_sink, is_sink


def check_size_distribution(sampler, sizes):
	values = jnp.arange(sampler.min_size, sampler.max_size + 1)
	weights = jnp.ones(values.shape) if sampler.p is None else sampler.p ** values
	expected = weights / jnp.sum(weights)
	observed = jnp.array([jnp.mean(sizes == v) for v in values])
	assert jnp.all(sizes >= sampler.min_size) and jnp.all(sizes <= sampler.max_size)
	assert jnp.all(jnp.abs(observed - expected) < 0.05), f"Size frequencies {observed} do not match {expected}."


def test_reach_sampler(max_size):
	key = KEY

	samplers = [
		ReachSampler(max_size=max_size),
		ReachSampler(n_tokens=5, max_size=max_size, p=0.5, prob_stutter=1.0),
	]

	for sampler in samplers:
		for i in range(N_SAMPLES):
			key, subkey1, subkey2 = jax.random.split(key, 3)
			dfax = sampler.sample(subkey1)
			check_sample(sampler, dfax)

			# A chain 0 -> 1 -> ... -> n - 1 whose only accepting state is its end, with no way to fail
			n = int(dfax.n_states)
			is_accept, is_reject_sink, is_sink = state_masks(dfax)
			assert jnp.array_equal(is_accept, jnp.arange(sampler.max_size) == n - 1)
			assert is_sink[n - 1] and not jnp.any(is_reject_sink)
			for s in range(n - 1):
				row = dfax.transitions[s]
				n_advance = int(jnp.sum(row == s + 1))
				assert jnp.all((row == s) | (row == s + 1)) and n_advance >= 1
				if sampler.prob_stutter == 1.0:
					assert n_advance == 1

			# Reaching acceptance takes the whole chain
			word = dfax.find_word(subkey2)
			assert int(jnp.sum(word != -1)) == n - 1
			assert dfax.advance(word).reward() == 1

			print(f"Test completed for {i + 1} samples.".ljust(50), end="\r")

		# The number of states is the sampled size
		key, subkey = jax.random.split(key)
		sizes = jax.vmap(sampler.sample)(jax.random.split(subkey, N_SAMPLES * 30)).is_reach.sum(axis=1)
		check_size_distribution(sampler, sizes)

		print(f"Finished running tests for {sampler}.")

	print("All tests passed for reach sampler.")


def test_reach_avoid_sampler(max_size):
	key = KEY

	samplers = [
		ReachAvoidSampler(max_size=max_size),
		ReachAvoidSampler(n_tokens=5, max_size=max_size, p=0.5, prob_stutter=1.0),
	]

	for sampler in samplers:
		for i in range(N_SAMPLES):
			key, subkey1, subkey2 = jax.random.split(key, 3)
			dfax = sampler.sample(subkey1)
			check_sample(sampler, dfax)

			# One accepting sink, one rejecting sink, and every other state can both progress and fail
			n = int(dfax.n_states)
			is_accept, is_reject_sink, is_sink = state_masks(dfax)
			assert int(jnp.sum(is_accept)) == 1 and jnp.all(is_sink[is_accept])
			assert int(jnp.sum(is_reject_sink)) == 1
			fail = int(jnp.argmax(is_reject_sink))
			for s in range(n):
				if is_sink[s]:
					continue
				row = dfax.transitions[s]
				n_fail = int(jnp.sum(row == fail))
				n_advance = int(jnp.sum((row != s) & (row != fail)))
				assert n_fail >= 1 and n_advance >= 1
				if sampler.prob_stutter == 1.0:
					assert n_fail == 1 and n_advance == 1

			# Reaching acceptance takes the whole chain, while failing takes a single step from the start
			word = dfax.find_word(subkey2)
			assert int(jnp.sum(word != -1)) == n - 2
			assert dfax.advance(word).reward() == 1
			assert any(dfax.advance(a).reward() == -1 for a in range(sampler.n_tokens))

			print(f"Test completed for {i + 1} samples.".ljust(50), end="\r")

		# The number of states is the sampled size
		key, subkey = jax.random.split(key)
		sizes = jax.vmap(sampler.sample)(jax.random.split(subkey, N_SAMPLES * 30)).is_reach.sum(axis=1)
		check_size_distribution(sampler, sizes)

		print(f"Finished running tests for {sampler}.")

	print("All tests passed for reach-avoid sampler.")


def test_rad_sampler(max_size):
	key = KEY

	samplers = [
		RADSampler(max_size=max_size),
		RADSampler(n_tokens=5, max_size=max_size, p=None, max_mutations=0),
	]

	for sampler in samplers:
		unmutated = sampler.replace(max_mutations=0)
		has_reject_sink = []
		n_mutated = 0

		for i in range(N_SAMPLES):
			key, subkey1, subkey2 = jax.random.split(key, 3)
			dfax = sampler.sample(subkey1)
			check_sample(sampler, dfax)

			# Non-trivial and satisfiable: the start is not accepting, and the only accepting state is a sink
			is_accept, is_reject_sink, is_sink = state_masks(dfax)
			assert 2 <= int(dfax.n_states) <= sampler.max_size
			assert not dfax.labels[dfax.start]
			assert int(jnp.sum(is_accept)) == 1 and jnp.all(is_sink[is_accept])
			assert int(jnp.sum(is_reject_sink)) <= 1
			word = dfax.find_word(subkey2)
			assert dfax.advance(word).reward() == 1

			has_reject_sink.append(bool(jnp.any(is_reject_sink)))
			n_mutated += int(not (dfax == unmutated.sample(subkey1)))

			print(f"Test completed for {i + 1} samples.".ljust(50), end="\r")

		# Avoid constraints are optional, so DFAs both with and without a rejecting sink show up
		assert any(has_reject_sink) and not all(has_reject_sink)
		# Mutations are applied on top of the same base DFA
		assert (n_mutated > 0) == (sampler.max_mutations > 0)

		print(f"Finished running tests for {sampler}.")

	print("All tests passed for RAD sampler.")


def test_mixture(max_size):
	key = KEY

	# (name, mixture, components it should reproduce in R, RA, RAD order)
	mixtures = [
		(
			"default",
			MixtureSampler(max_size=max_size),
			# Class defaults give the expected min sizes (2, 3, 3); p is shared, so RAD gets None too
			[ReachSampler(max_size=max_size, p=None), ReachAvoidSampler(max_size=max_size, p=None), RADSampler(max_size=max_size, p=None)]
		),
		(
			"custom",
			MixtureSampler(n_tokens=5, max_size=max_size, p=0.5, prob_stutter=0.5, max_mutations=2, min_size=(3, 4, 4), probs=(2, 1, 1)),
			[
				ReachSampler(n_tokens=5, max_size=max_size, p=0.5, prob_stutter=0.5, min_size=3),
				ReachAvoidSampler(n_tokens=5, max_size=max_size, p=0.5, prob_stutter=0.5, min_size=4),
				RADSampler(n_tokens=5, max_size=max_size, p=0.5, prob_stutter=0.5, min_size=4, max_mutations=2)
			]
		),
		(
			"RA only",
			MixtureSampler(max_size=max_size, probs=(0, 1, 0)),
			[ReachSampler(max_size=max_size, p=None), ReachAvoidSampler(max_size=max_size, p=None), RADSampler(max_size=max_size, p=None)]
		),
	]

	for name, mixture, components in mixtures:
		probs = jnp.array(mixture.probs) / jnp.sum(jnp.array(mixture.probs))
		sample_keys, dfaxs = [], []

		for i in range(N_SAMPLES):
			key, subkey = jax.random.split(key)
			dfax = mixture.sample(subkey)
			check_sample(mixture, dfax)

			# The mixture uses the first half of the split for the component and the second to choose it
			component_key, choice_key = jax.random.split(subkey)
			idx = jax.random.choice(choice_key, len(components), p=probs)
			assert probs[idx] > 0
			assert dfax == components[idx].sample(component_key), f"Mixture sample does not match component {int(idx)} for sample {i + 1}."

			sample_keys.append(subkey)
			dfaxs.append(dfax)

			print(f"Test completed for {i + 1} samples ({name} mixture).".ljust(50), end="\r")

		# Sampling under vmap gives the same DFAs
		vmap_dfaxs = jax.vmap(mixture.sample)(jnp.stack(sample_keys))
		seq_dfaxs = jax.tree.map(lambda *xs: jnp.stack(xs), *dfaxs)
		assert jnp.all(jax.vmap(lambda a, b: a == b)(vmap_dfaxs, seq_dfaxs))

		# Components are chosen with the given (normalized) probabilities
		key, subkey = jax.random.split(key)
		choice_keys = jax.vmap(lambda k: jax.random.split(k)[1])(jax.random.split(subkey, N_SAMPLES * 30))
		idxs = jax.vmap(lambda k: jax.random.choice(k, len(components), p=probs))(choice_keys)
		freqs = jnp.array([jnp.mean(idxs == j) for j in range(len(components))])
		assert jnp.all(jnp.abs(freqs - probs) < 0.05), f"Component frequencies {freqs} do not match probabilities {probs}."

		print(f"Finished running tests for {name} mixture.".ljust(50))

	try:
		MixtureSampler(max_size=max_size).sample(KEY, accept_set=jnp.zeros((1, 1), dtype=jnp.int32))
		assert False, "Conditioned sampling should not be supported by MixtureSampler."
	except NotImplementedError:
		pass

	print("All tests passed for mixture sampler.")


def test_data_sampler():
	key = KEY

	# Tight DFAs of different sizes, as in the bundled datasets, plus a single-state rejecting DFA
	samplers = [ReachSampler(max_size=5), ReachAvoidSampler(max_size=5)]
	data = {}
	for i in range(20):
		key, subkey = jax.random.split(key)
		data[samplers[i % 2].sample(subkey).shrink()] = (f"DFA {i}", [float(i)] * 4)
	data[samplers[0].trivial(False).shrink()] = ("Failure.", [-1.0] * 4)

	dfax_list = list(data.keys())
	max_n = max(d.max_n_states for d in dfax_list)
	assert len({d.max_n_states for d in dfax_list}) > 1

	def accepts(dfax, word):
		return bool(dfax.advance(word).reward() == 1)

	for name, sampler in [("list", data2sampler(dfax_list)), ("dict", data2sampler(data))]:
		# Entries are stored in order, expanded to the largest size
		assert sampler.dfax_array.transitions.shape == (len(dfax_list), max_n, dfax_list[0].n_tokens)
		entries = [jax.tree.map(lambda x: x[j], sampler.dfax_array) for j in range(len(dfax_list))]
		assert all(d == entry for d, entry in zip(dfax_list, entries))
		if name == "dict":
			assert all(jnp.allclose(sampler.embed(entry), jnp.array(data[d][1])) for d, entry in zip(dfax_list, entries))
		else:
			assert sampler.embd_array is None

		for label, reward in [(True, 1), (False, -1)]:
			trivial = sampler.trivial(label)
			assert all(a.shape == b.shape for a, b in zip(jax.tree.leaves(trivial), jax.tree.leaves(entries[0])))
			assert trivial.reward() == reward

		def index_of(dfax):
			match = (sampler.dfax_array.start == dfax.start) \
				& jnp.all(sampler.dfax_array.transitions == dfax.transitions, axis=(1, 2)) \
				& jnp.all(sampler.dfax_array.labels == dfax.labels, axis=1)
			return jnp.argmax(match), jnp.sum(match)

		# Unconditioned samples are dataset entries drawn uniformly
		key, subkey = jax.random.split(key)
		idxs, n_matches = jax.vmap(lambda k: index_of(sampler.sample(k)))(jax.random.split(subkey, N_SAMPLES * 30))
		assert jnp.all(n_matches == 1)
		freqs = jnp.bincount(idxs, length=len(entries)) / idxs.size
		assert jnp.all(jnp.abs(freqs - 1 / len(entries)) < 0.02), f"Entry frequencies {freqs} are not uniform."

		# Conditioned samples are drawn from the entries accepting the accept set and rejecting the reject set
		key, subkey = jax.random.split(key)
		target = entries[0]
		accept_word = target.find_word(subkey)
		assert accepts(target, accept_word)
		words = [e.find_word(subkey) for e in entries[1:]]
		reject_word = next(w for e, w in zip(entries[1:], words) if accepts(e, w) and not accepts(target, w))
		accept_set, reject_set = accept_word[None], reject_word[None]

		valid = {j for j, e in enumerate(entries) if accepts(e, accept_word) and not accepts(e, reject_word)}
		assert 0 in valid
		sampled = set()
		for i in range(N_SAMPLES):
			key, subkey = jax.random.split(key)
			dfax = sampler.sample(subkey, accept_set=accept_set, reject_set=reject_set)
			assert accepts(dfax, accept_word) and not accepts(dfax, reject_word)
			idx, n_match = index_of(dfax)
			assert int(n_match) == 1 and int(idx) in valid
			sampled.add(int(idx))

			print(f"Test completed for {i + 1} samples ({name} data).".ljust(50), end="\r")

		assert sampled == valid

		# Unsatisfiable constraints fall back to the single-state rejecting DFA
		dfax = sampler.sample(subkey, accept_set=accept_set, reject_set=accept_set)
		assert int(dfax.n_states) == 1 and dfax.reward() == -1

		print(f"Finished running tests for {name} data.".ljust(50))

	print("All tests passed for data sampler.")


def main():
	sampler = ReachAvoidSampler(max_size=5, p=None)
	dfa_sizes = [5, 10, 20]
	expand_size = 50

	test_prompt_and_advance(sampler)
	test_binary_ops(sampler)
	test_find_word(sampler)
	test_shrink(dfa_sizes)
	test_expand(sampler, expand_size)
	test_reach_sampler(sampler.max_size)
	test_reach_avoid_sampler(sampler.max_size)
	test_rad_sampler(sampler.max_size)
	test_mixture(sampler.max_size)
	test_data_sampler()


if __name__ == '__main__':
	main()
