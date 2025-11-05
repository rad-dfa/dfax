# DFAx

A JAX-compatible Python implementation of a Deterministic Finite Automaton (DFA).

## Installation

This package will soon be made pip-installable. In the meantime, pull the repo and and install locally.

```
git clone https://github.com/rad-dfa/dfax.git
pip install -e dfax
```

## Usage

Create DFAs by specifying a `start` state, `transitions` matrix, which is max number of states by number of alphabet symbols, and the associated `labels` for each state.

```python
from dfax import DFAx

dfax = DFAx(
	start=0, # State referred to as 0 is the initial state
	transitions=jnp.array([
		[1, 2, 0, 0, 0],
		[1, 1, 1, 1, 1],
		[2, 2, 2, 2, 2],
	]), # Max number of states is 3 and number of tokens is 5
	labels=jnp.array([False, True, False]) # State labels
) # Returns a DFA
```

Take transitions on the DFA using a given symbol.

```python
dfax = dfax.advance(0) # Returns the resulting DFA after reading the symbol referred to as 0
```

Minimize DFAs.

```python
dfax = dfax.minimize() # Returns a canonical minimal DFA
```


Canonicalize DFAs by relabeling states based on a BFS search.

```python
dfax = dfax.canonicalize() # Returns a canonical DFA
```

Mutate DFAs by randomly toggling entries in the transition matrix.

```python
import jax

key = jax.random.PRNGKey(0)
dfax = dfax.mutate(key) # Returns a mutated DFA
```

Perform syntactic equality check between DFAs.

```python
dfax1 == dfax2
```

Perform semantic equality check between DFAs.

```python
dfax1.minimize() == dfax2.minimize()
```

Use DFAs as reward functions. With ternary semantics, reward is (i) `+1` if the `start` state has label `True`, (ii) `-1` if the `start` state has label `False` and is a sink state, and (iii) `0` otherwise. With binary semantics, `0` is returned instead of `-1`.

```python
dfax.reward() # Returns a ternary reward
dfax.reward(binary=True) # Returns a binary reward
```


Sample from different DFA distributions: `Reach` samples DFAs ordering alphabet symbols, `ReachAvoid` samples `Reach` DFAs but also includes `Avoid` constraints, and `ReachAvoidDerived` samples randomly mutated `Reach` and `ReachAvoid` DFAs.

```python
import jax
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler

key = jax.random.PRNGKey(0)
sampler = ReachAvoidSampler()

dfax = sampler.sample(key)
```


Define your own DFA samplers by overloading `DFASampler `.

```python
@struct.dataclass
class MySampler(DFASampler):
    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key: chex.PRNGKey) -> DFAx:
        # Write sampling code and return sampled DFA
```

Visualize DFAs.

```python
from dfax.utils import visualize
visualize(dfax)
```


This project is a JAX extension of [dfa](https://github.com/mvcisback/dfa). Therefore, we include helper methods for translating `DFAx` objects to and from `DFA` objects.

```python
from dfax import dfa2dfax, dfax2dfa

dfa = dfax2dfa(dfax) # Create DFA from DFAx
dfax = dfa2dfax(dfa) # Create DFAx from DFA
```

## In progress

Currently, we are working on implementing Boolean operations on `DFAx` objects, e.g., conjunction, disjunction, etc. If there are other functionalities you would like to have in this package, create pull request or contact us to work together!





