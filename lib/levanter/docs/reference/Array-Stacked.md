# ArrayStacked

[levanter.layers.stacked.ArrayStacked][] is an array-native stack container for scan-over-layers when you are not using named axes in
the stack itself.

The container uses ordinary JAX arrays and Equinox modules. Import it from
`levanter.layers.stacked`; the package does not import Haliax.

## When to use `ArrayStacked`

Use `ArrayStacked` when:

* You want scan-over-layers behavior (`fold`, `scan`, `fold_via`, `scan_via`) with plain `jax.Array` leaves.
* You want layer parameters stored as arrays with a leading layer dimension.
* You do not want to thread a named `Axis` for the layer stack.
* You want to avoid unrolled Python layer loops that can increase compile time and HBM pressure.

Compared to [haliax.nn.Stacked][]:

* `Stacked` uses a named block axis (`Block: Axis`) and can stack `NamedArray` leaves on that axis.
* `ArrayStacked` uses `num_layers: int` and stacks array leaves on positional axis `0`.

## Compile Time and Memory Notes

`ArrayStacked` keeps the scan-over-layers structure explicit (via `lax.scan`) instead of relying on Python-unrolled layer
loops. In practice, this often:

* Reduces compile time.
* Reduces compile-time/live-range memory pressure (which can help avoid compile OOMs).
* Sometimes improves step-time throughput when the unrolled alternative also compiles.

As always, validate with your model, mesh, and batch/sequence settings.

## Initialization

`ArrayStacked.init(num_layers, Module)` returns an initializer:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from levanter.layers.stacked import ArrayStacked


class Block(eqx.Module):
    weight: jax.Array

    @staticmethod
    def init(weight):
        return Block(weight=weight)

    def __call__(self, carry, layer_scale, mask):
        return carry + self.weight * layer_scale + jnp.sum(mask)


num_layers = 4
width = 8

stack = ArrayStacked.init(num_layers, Block)(
    weight=jax.random.normal(jax.random.PRNGKey(0), (num_layers, width))
)
```

An array init argument is sliced per layer when it has at least one dimension
and `shape[0] == num_layers`. Other arguments, including config objects, are
shared. Initialization has no `in_axes` override: wrap an accidentally matching
shared array in a closure inside the initializer if it must remain shared.

All layers are built in a single mapped trace, so `Module.init` must be traceable: it may branch in
Python on a shared argument such as a config field, but not on a per-layer value. `scan` already
requires the same of `__call__`, since one compiled body runs every layer, so per-layer variation
belongs in an argument of leading size `num_layers`. A non-array value the module returns is kept
as that value rather than stacked, so a non-static `int` field reads back as an `int`.

## Fold / Scan Usage

### `fold` / `fold_via`

`fold` calls each layer’s `__call__` and returns only the final carry.
`stack.fold(carry0, layer_scale, mask, in_axes=(0, None))` uses the `Block` above.
Use `fold_via` to select another layer function:

```python
carry0 = jnp.zeros((width,))
layer_scale = jnp.array([1.0, 2.0, 3.0, 4.0])
mask = jnp.ones((16, 16))

# in_axes follows vmap-like semantics for non-carry arguments.
# layer_scale is scanned on axis 0; mask is shared across layers.
carry = stack.fold_via(Block.__call__, in_axes=(0, None))(carry0, layer_scale, mask)
```

### `scan` / `scan_via`

`scan` calls each layer’s `__call__`, which must return `(carry, output)`.
`scan_via` selects a separate function, as in this example:

```python
def block_scan(layer: Block, carry, x):
    carry = carry + layer.weight * x
    return carry, carry

carry, per_layer = stack.scan_via(block_scan, in_axes=0)(jnp.zeros((width,)), jnp.ones((num_layers, width)))
```

## `in_axes` semantics

`ArrayStacked` uses a vmap-like convention for non-carry inputs:

* `in_axes=None` (default): all extra args/kwargs are shared across layers.
* `in_axes=0`: all positional extra arguments are sliced on axis 0; keyword
  arguments remain shared.
* A positional tuple such as `(0, None)` selects an axis for each positional
  argument. Keywords remain shared.
* `in_axes=(args_axes, kwargs_axes)` (identified by a dictionary in its second slot):
  `args_axes` is a tuple aligned to positional args;
  `kwargs_axes` is a dict covering every supplied keyword exactly. For example,
  `in_axes=((0,), {"mask": None})` scans one positional argument and shares `mask`.

Any argument with an integer axis must be a `jax.Array` with `shape[axis] == num_layers`.

## Other helpers

* `get_layer(i)`: returns layer `i` as a regular module instance.
* `unstacked()`: returns a tuple of per-layer modules.
* `vmap_via(fn, in_axes=..., out_axes=...)`: applies `fn(layer, ...)` independently to each layer and stacks outputs.

## Checkpointing

Pass `gradient_checkpointing` to `ArrayStacked.init`; it defaults to `False`.
For example, `ArrayStacked.init(num_layers, Block, gradient_checkpointing="nested")`
returns an initializer that uses nested rematerialization. The setting accepts a
boolean, a shorthand string, or
[haliax.ScanCheckpointPolicy][]. `True` checkpoints
each block, `False` disables rematerialization, `"full"` recomputes carries during
the backward pass, `"nested"` checkpoints groups of layers for backward-pass
recomputation, and `"offload"` saves carries and inputs
in pinned host memory. Offloading requires backend support.

Parameter arrays retain a leading layer dimension under the `stacked` field.
`to_state_dict` exports separate `<prefix>.<layer>.<parameter>` entries;
`from_state_dict` restores them into the existing parameter structure.

Use an explicit JAX mesh and native parameter shardings. Sharding is independent
of layer count and shape names; the stack does not infer distribution from a
logical-axis mapping.

::: haliax.ScanCheckpointPolicy


## Native Llama

[levanter.models.llama_native.NativeLlamaModel][] uses `ArrayStacked`, ordinary
JAX arrays and explicit mesh axes. It remains separate from the existing
`levanter.models.llama` implementation while numerical parity, placement and
training performance are compared on the same parameters.

This example requires four devices. For a CPU smoke, start Python with
`JAX_NUM_CPU_DEVICES=4`.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from levanter.models.llama_native import NativeLlamaConfig, NativeLlamaSharding

mesh = Mesh(
    np.asarray(jax.devices()[:4]).reshape(2, 2),
    ("data", "model"),
    axis_types=(AxisType.Explicit, AxisType.Explicit),
)
layout = NativeLlamaSharding(
    batch_axes=("data",), fsdp_axis="data", model_axis="model"
)
config = NativeLlamaConfig(
    vocab_size=128, hidden_dim=64, intermediate_dim=128,
    num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=16,
    attention_implementation="reference", gradient_checkpointing=True,
)

with jax.set_mesh(mesh):
    model = config.build(key=jax.random.key(0), sharding=layout)
    tokens = jnp.zeros((4, 16), dtype=jnp.int32)
    weights = jnp.ones((4, 16), dtype=jnp.float32)
    loss = eqx.filter_jit(
        lambda m: m.next_token_loss(tokens, weights, loss_implementation="xla")
    )(model)
```

The layout separates stored parameters from their forward computation.
Here `P` denotes `jax.sharding.PartitionSpec`, and projection weights use
`[output_features, input_features]` order. Column-projection weights have storage `P("model", "data")` and compute
`P("model", None)`: forward gathers the FSDP input dimension while keeping
output channels partitioned across `model`. Row projections use
`P("data", "model")` in storage and `P(None, "model")` for computation.
Residual activations partition the batch across `data`; attention heads, MLP
intermediates and output vocabulary partition across `model`. The layer axis
is replicated. Initialization and forward calls require a `jax.set_mesh`
context.

`next_token_loss` uses the tokens as inputs and shifts them to form targets.
A weight at position `t` controls the loss for predicting token `t + 1`;
the final position always has zero weight.

The current model covers dense SwiGLU Llama with grouped-query attention,
default or linear-scaled RoPE, structured causal/segment masks and explicit
position IDs. Select raw-array attention kernels with
`attention_implementation`; `reference` is the default, and accelerated
implementations retain their backend and shape requirements. `xla_flash` is
rejected because its current implementation uses named-array operations.
Paged decoding and MoE are outside this model's scope.

Use `model = model.from_state_dict(state)` outside the compiled training step
to import an existing Llama state dict. It returns a model with the imported
parameters in the template model's dtype and storage layout. The input uses
canonical HF keys and raw arrays; articulated projection dimensions from the
existing model's `to_state_dict()` are also accepted. `model.to_state_dict()`
exports output-first projection arrays under the same keys. Calling it on the
model tree returned by differentiation exports parameter gradients.

::: levanter.models.llama_native.NativeLlamaModel
    options:
      members: false

## API

::: levanter.layers.stacked.ArrayStacked
