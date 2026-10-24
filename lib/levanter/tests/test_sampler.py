# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import haliax as hax
from levanter.layers.sampler import Sampler


def test_sampler_top_p_keeps_only_the_nucleus_head():
    vocab = hax.Axis("vocab", 4)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.array([6.0, 5.0, 1.0, -1.0], dtype=jnp.float32), (vocab,))

    token, log_prob = sampler(
        logits,
        jnp.array(1.0, dtype=jnp.float32),
        top_ps=jnp.array(0.5, dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
    )

    assert int(token.array) == 0
    assert float(log_prob.array) == pytest.approx(0.0)


def test_sampler_top_p_keeps_cutoff_crossing_token():
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.log(jnp.array([0.4, 0.35, 0.25], dtype=jnp.float32)), (vocab,))

    masked_logits = sampler._apply_top_p(logits, jnp.array(0.6, dtype=jnp.float32))

    assert jnp.isfinite(masked_logits.array[:2]).all()
    assert jnp.isneginf(masked_logits.array[2])


def test_sampler_top_p_does_not_overshoot_exact_threshold():
    vocab = hax.Axis("vocab", 3)
    sampler = Sampler(vocab)
    logits = hax.named(jnp.log(jnp.array([0.4, 0.35, 0.25], dtype=jnp.float32)), (vocab,))

    masked_logits = sampler._apply_top_p(logits, jnp.array(0.4, dtype=jnp.float32))

    assert jnp.isfinite(masked_logits.array[0])
    assert jnp.isneginf(masked_logits.array[1:]).all()


@pytest.mark.tpu
@pytest.mark.parametrize(
    ("axis_type", "top_p"),
    [
        (AxisType.Auto, 0.4),
        (AxisType.Explicit, 0.4),
        (AxisType.Auto, 0.6),
        (AxisType.Explicit, 0.6),
        (AxisType.Auto, None),
        (AxisType.Explicit, None),
    ],
    ids=[
        "auto-exact-cutoff",
        "explicit-exact-cutoff",
        "auto-crossing-cutoff",
        "explicit-crossing-cutoff",
        "auto-no-top-p",
        "explicit-no-top-p",
    ],
)
@pytest.mark.parametrize("temperature", [1.0, 0.0, (0.0, 0.5, 1.0, 2.0)], ids=["stochastic", "greedy", "mixed"])
def test_sampler_sharded_vocab_matches_replicated_tokens_and_logprobs(axis_type, top_p, temperature):
    model_size = 2 if len(jax.devices()) >= 4 else 1
    mesh = Mesh(np.asarray(jax.devices()).reshape(-1, model_size), ("data", "model"), axis_types=(axis_type,) * 2)
    batch_size = 2 * mesh.shape["data"]
    batch, vocab = hax.Axis("batch", batch_size), hax.Axis("vocab", 4)
    probabilities = np.tile(np.asarray([[0.4, 0.35, 0.2, 0.05]], dtype=np.float32), (batch_size, 1))
    temperature_values = np.resize(np.asarray(temperature, dtype=np.float32), batch_size)
    logits_array = np.log(probabilities)
    if top_p is None:
        # An unsharded JAX oracle preserves the global key and complete token
        # domain independently of the sampler's input/output placement.
        scaled = jnp.asarray(logits_array / np.where(temperature_values == 0, 1.0, temperature_values)[:, None])
        reference_tokens = np.where(
            temperature_values == 0,
            np.argmax(logits_array, axis=-1),
            np.asarray(jax.random.categorical(jax.random.PRNGKey(7), scaled)),
        )
        reference_logprobs = np.asarray(jax.nn.log_softmax(scaled))[np.arange(batch_size), reference_tokens]
    with jax.set_mesh(mesh), hax.axis_mapping({"batch": "data", "vocab": "model"}):
        sharded = hax.named(jax.device_put(logits_array, NamedSharding(mesh, P("data", "model"))), (batch, vocab))
        replicated = hax.named(jax.device_put(logits_array, NamedSharding(mesh, P("data", None))), (batch, vocab))
        temperatures = hax.named(jax.device_put(temperature_values, NamedSharding(mesh, P("data"))), (batch,))
        sampler = Sampler(vocab)
        sample = jax.jit(lambda logits: sampler(logits, temperatures, top_ps=top_p, key=jax.random.PRNGKey(7)))
        expected_tokens, expected_logprobs = sample(replicated)
        tokens, logprobs = sample(sharded)
        np.testing.assert_array_equal(tokens.array, expected_tokens.array)
        np.testing.assert_allclose(logprobs.array, expected_logprobs.array, atol=1e-6, rtol=1e-6)
        if top_p is None:
            np.testing.assert_array_equal(tokens.array, reference_tokens)
            np.testing.assert_allclose(logprobs.array, reference_logprobs, atol=1e-6, rtol=1e-6)
        if mesh.shape["data"] > 1:
            assert tokens.array.sharding.spec == P("data")
            assert logprobs.array.sharding.spec == P("data")
        if temperature == 1.0:
            if top_p == 0.4:
                np.testing.assert_array_equal(tokens.array, 0)
            elif top_p == 0.6:
                assert np.isin(np.asarray(tokens.array), [0, 1]).all()
