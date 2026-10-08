from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reasoning_gym
from flax import serialization

from rl2 import train_grpo_reasoning as grpo
from rl2.gdn2 import GatedDeltaNet2Config, GatedDeltaNet2LM
from rl2.gdn2.checkpoints import Parameters, load_checkpoint, save_checkpoint


@pytest.fixture(scope="module")
def tiny_model() -> tuple[GatedDeltaNet2LM, Parameters]:
    model = GatedDeltaNet2LM(GatedDeltaNet2Config(hidden_size=2, head_dim=2, num_heads=1, conv_size=2), 1, 3, 5)
    variables = model.init(jax.random.key(1), jnp.array([[1, 3]], jnp.int32), jnp.array([2], jnp.int32))
    return model, variables["params"]


def make_batch() -> grpo.GRPOBatch:
    return grpo.GRPOBatch(
        jnp.array([[1, 3, 0], [1, 4, 3]], jnp.int32),
        jnp.array([2, 3], jnp.int32),
        jnp.array([[3, 2, 0], [4, 3, 2]], jnp.int32),
        jnp.array([[True, True, False], [True, True, True]]),
        jnp.zeros((2, 3), jnp.float32),
        jnp.zeros((2, 3), jnp.float32),
        jnp.array([1, -1], jnp.float32),
    )


def test_cached_sampling_matches_packed_replay_and_eos(tiny_model: tuple[GatedDeltaNet2LM, Parameters]) -> None:
    model, params = tiny_model
    batch = make_batch()
    key = jax.random.key(8)
    args = {"group_size": 2, "max_new_tokens": 3, "temperature": 0.7}
    tokens, _, _, _ = grpo.generate(model, params, batch.prompts, batch.prompt_lengths, key, eos_token_id=-1, **args)
    eos = int(tokens[0, 0])
    tokens, old, mask, next_key = grpo.generate(
        model,
        params,
        batch.prompts,
        batch.prompt_lengths,
        key,
        eos_token_id=eos,
        **args,
    )
    assert mask[0].tolist() == [True, False, False]
    assert tokens[0].tolist() == [eos, 0, 0]
    assert old[0, 1:].tolist() == [0, 0]
    repeated = grpo.GRPOBatch(
        jnp.repeat(batch.prompts, 2, axis=0),
        jnp.repeat(batch.prompt_lengths, 2),
        tokens,
        mask,
        old,
        jnp.zeros_like(old),
        jnp.zeros(4, jnp.float32),
    )
    replay = grpo.completion_log_probs(model, params, repeated, 0.7)
    np.testing.assert_allclose(replay, old, atol=2e-6)
    again = grpo.generate(model, params, batch.prompts, batch.prompt_lengths, key, eos_token_id=eos, **args)
    for actual, expected in zip(again[:3], (tokens, old, mask), strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(jax.random.key_data(again[3]), jax.random.key_data(next_key))
    assert not np.array_equal(jax.random.key_data(key), jax.random.key_data(next_key))


def test_replay_matches_independent_unpadded_sequences(tiny_model: tuple[GatedDeltaNet2LM, Parameters]) -> None:
    model, params = tiny_model
    batch = make_batch()
    actual = grpo.completion_log_probs(model, params, batch, 0.7)
    for i in range(2):
        prompt_length = int(batch.prompt_lengths[i])
        completion_length = int(batch.mask[i].sum())
        ids = jnp.concatenate((batch.prompts[i, :prompt_length], batch.completions[i, :completion_length]))
        _, logits = model.apply({"params": params}, ids[None, :-1], jnp.array([len(ids) - 1], jnp.int32))
        expected = grpo.policy_log_probs(
            logits[:, prompt_length - 1 :], batch.completions[i : i + 1, :completion_length], 0.7
        )
        np.testing.assert_allclose(actual[i, :completion_length], expected[0], atol=2e-6)
    assert actual[0, -1] == 0


def test_group_advantages_and_clipped_sequence_weighting() -> None:
    rewards = jnp.array([[0, 1], [0.5, 0.5]], jnp.float32)
    advantages = grpo.group_advantages(rewards)
    np.testing.assert_allclose(advantages[0], [-1, 1], atol=3e-4)
    np.testing.assert_array_equal(advantages[1], [0, 0])
    batch = make_batch()
    # Equal sequence weights despite different completion lengths. Both signs clip.
    ratios = jnp.array([[1.5, 1, 99], [0.5, 1, 1]], jnp.float32)
    loss, metrics = grpo.objective(jnp.log(ratios), batch, clip_coef=0.2, beta=0)
    expected = -((1.2 + 1) / 2 - (0.8 + 1 + 1) / 3) / 2
    np.testing.assert_allclose(loss, expected, atol=1e-6)
    np.testing.assert_allclose(metrics["policy/clip_fraction"], (1 / 2 + 1 / 3) / 2)

    def loss_fn(log_probs: jax.Array) -> jax.Array:
        return grpo.objective(log_probs, batch, 0.2, 0)[0]

    grad = jax.grad(loss_fn)(jnp.log(ratios))
    assert grad[0, 0] == 0 and grad[1, 0] == 0 and grad[0, 2] == 0
    assert grad[0, 1] < 0 and grad[1, 1] > 0
    # KL still trains equal-reward groups, while beta=0 leaves them unchanged.
    batch = batch._replace(advantages=jnp.zeros(2, jnp.float32), ref_log_probs=jnp.full((2, 3), -1.0))
    loss, metrics = grpo.objective(jnp.zeros((2, 3), jnp.float32), batch, 0.2, 0.1)
    np.testing.assert_allclose(loss, 0.1 * np.exp(-1), atol=1e-6)
    assert metrics["policy/ref_kl"] > 0


def test_update_and_resumable_checkpoint(
    tiny_model: tuple[GatedDeltaNet2LM, Parameters],
    tmp_path: Path,
) -> None:
    model, params = tiny_model
    config = grpo.Config(num_tasks=1, group_size=2, num_minibatches=1, learning_rate=0.001)
    state = grpo.create_state(model, params, config)
    batch = make_batch()
    old = grpo.completion_log_probs(model, params, batch, config.temperature)
    batch = batch._replace(old_log_probs=old, ref_log_probs=old)
    updated, metrics = grpo.update(
        state,
        model,
        batch,
        temperature=config.temperature,
        clip_coef=config.clip_coef,
        beta=config.beta,
    )
    assert int(updated.step) == 1
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(updated.params), strict=True)
    )
    key = jax.random.key(42)
    grpo.save_training_checkpoint(str(tmp_path), updated, key, 3, config)
    data = (tmp_path / "checkpoint.msgpack").read_bytes()
    restored, restored_key, iteration = grpo.restore_training_checkpoint(
        data, state, replace(config, total_updates=2000)
    )
    assert iteration == 3
    for a, b in zip(jax.tree.leaves(updated), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(restored_key))
    # Restored optimizer moments produce the same next update.
    next_state, _ = grpo.update(updated, model, batch, temperature=1.0, clip_coef=0.2, beta=0.01)
    next_restored, _ = grpo.update(restored, model, batch, temperature=1.0, clip_coef=0.2, beta=0.01)
    for a, b in zip(jax.tree.leaves(next_state), jax.tree.leaves(next_restored), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="configuration differs"):
        grpo.restore_training_checkpoint(data, state, replace(config, temperature=0.5))
    payload = serialization.msgpack_restore(data)
    payload["version"] = 999
    with pytest.raises(ValueError, match="version"):
        grpo.restore_training_checkpoint(serialization.msgpack_serialize(payload), state, config)
    save_checkpoint(tmp_path / "export", model, jax.device_get(updated.params), {})
    _, loaded = load_checkpoint(tmp_path / "export")
    for a, b in zip(jax.tree.leaves(updated.params), jax.tree.leaves(loaded["params"]), strict=True):
        np.testing.assert_array_equal(a, b)


class FakeTokenizer:
    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2

    def vocab_size(self) -> int:
        return 5

    def encode(self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool) -> list[int]:
        assert "<answer>" in text and "Question:" in text
        assert add_bos and not add_eos
        return [1, 3]

    def decode(self, ids: list[int]) -> str:
        assert 2 not in ids
        return "<answer>4</answer>" if ids else ""


def test_prompts_and_real_reasoning_gym_rewards() -> None:
    dataset = reasoning_gym.create_dataset("leg_counting", seed=1, size=2)
    entries = [dataset[i] for i in range(2)]
    tokens, lengths = grpo.encode_prompts(entries, FakeTokenizer(), 3)
    np.testing.assert_array_equal(tokens, [[1, 3, 0], [1, 3, 0]])
    np.testing.assert_array_equal(lengths, [2, 2])
    with pytest.raises(ValueError, match="Increase max_prompt_tokens"):
        grpo.encode_prompts(entries, FakeTokenizer(), 1)
    texts = [
        f"work\n<answer>wrong</answer><answer>{entries[0]['answer']}</answer>",
        str(entries[0]["answer"]),  # Correct raw answer still lacks required answer tags.
        f"<answer>\n{entries[1]['answer']}\n</answer>",
        "<answer>wrong</answer>",
    ]
    rewards = grpo.score_completions(dataset, entries, texts, 2)
    np.testing.assert_array_equal(rewards, [1, 0, 1, 0])
    with pytest.raises(ValueError, match="invalid reward"):
        grpo.score_completions(Mock(score_answer=Mock(return_value=float("nan"))), entries[:1], texts[:1], 1)


def test_format_sample_preserves_literal_multiline_text() -> None:
    prompt = "Question: first line\n\nsecond line\nAnswer:"
    completion = "thinking\n<answer>4</answer>\n```\n### Prompt\n```"
    sample = grpo.format_sample(prompt, completion, 0.5)
    assert "### Prompt\n\n    Question: first line\n    \n    second line\n    Answer:" in sample
    assert "### Completion\n\n    thinking\n    <answer>4</answer>\n    ```\n    ### Prompt\n    ```" in sample
    assert sample.endswith("**Reward:** 0.500")
    empty = grpo.format_sample(prompt, "", 0.0)
    assert "### Completion\n\n    \n\n**Reward:** 0.000" in empty


def test_collect_rollout_grouping_and_reference(
    tiny_model: tuple[GatedDeltaNet2LM, Parameters],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params = tiny_model
    config = grpo.Config(num_tasks=1, group_size=2, num_minibatches=1, max_prompt_tokens=3, max_new_tokens=3)
    state = grpo.create_state(model, params, config)
    entry = {"question": "How many legs?", "answer": "4"}
    dataset = Mock()
    dataset.__getitem__ = Mock(return_value=entry)
    dataset.score_answer = Mock(side_effect=[1.0, 0.0])
    tokenizer = Mock(wraps=FakeTokenizer())
    batch, diagnostics, samples, key = grpo.collect_rollout(
        state,
        model,
        params,
        dataset,
        tokenizer,
        jax.random.key(9),
        3,
        config,
    )
    dataset.__getitem__.assert_called_once_with(3)
    np.testing.assert_array_equal(batch.prompts, [[1, 3, 0], [1, 3, 0]])
    np.testing.assert_allclose(batch.old_log_probs, batch.ref_log_probs, atol=2e-6)
    assert len(samples) == 2
    encoded_prompt = tokenizer.encode.call_args.args[0]
    assert "### Prompt\n\n    " + encoded_prompt.replace("\n", "\n    ") in samples[0]
    assert "### Completion\n\n    <answer>4</answer>" in samples[0]
    assert samples[0].endswith("**Reward:** 1.000")
    assert samples[1].endswith("**Reward:** 0.000")
    assert diagnostics["charts/informative_group_fraction"] == 1.0
    np.testing.assert_allclose(batch.advantages, [1, -1], atol=3e-4)
    assert jax.random.key_data(key).shape == (2,)
    # Disabling KL must bypass the reference model entirely.
    monkeypatch.setattr(grpo, "completion_log_probs", Mock(side_effect=AssertionError("reference replay")))
    dataset.score_answer.side_effect = None
    dataset.score_answer.return_value = 0.0
    batch, _, _, _ = grpo.collect_rollout(
        state,
        model,
        {},
        dataset,
        FakeTokenizer(),
        jax.random.key(9),
        0,
        replace(config, beta=0),
    )
    np.testing.assert_array_equal(batch.ref_log_probs, np.zeros((2, 3)))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"group_size": 1}, "group_size"),
        ({"num_minibatches": 3}, "num_minibatches"),
        ({"temperature": 0}, "temperature"),
        ({"beta": float("nan")}, "beta"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"clip_coef": 1}, "clip_coef"),
        ({"seed": -1}, "seed"),
        ({"task_config": {"size": 1}}, "task_config"),
        ({"run_id": "../run"}, "run_id"),
    ],
)
def test_config_validation(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        grpo.Config(**overrides)


def test_yaml_config(tmp_path: Path) -> None:
    config = grpo.load_config("configs/grpo_reasoning.yaml")
    assert config.task == "leg_counting"
    assert config.run_id is not None
    assert config.checkpoint_interval_seconds == 600
    path = tmp_path / "config.yaml"
    path.write_text("group_size: 2\nnum_minibatches: 1\ntask_config:\n  max_animals: 3\n")
    assert grpo.load_config(path).task_config == {"max_animals": 3}
    path.write_text("group_size: 2.5\n")
    with pytest.raises(ValueError, match="type for group_size"):
        grpo.load_config(path)
    path.write_text("unknown: 1\n")
    with pytest.raises(ValueError, match="Unknown config"):
        grpo.load_config(path)


def test_training_loop_resume_and_logging_with_mocked_learning(
    tiny_model: tuple[GatedDeltaNet2LM, Parameters],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    model, params = tiny_model
    config = grpo.Config(
        log_dir=str(tmp_path),
        run_id="test",
        total_updates=4,
        num_tasks=1,
        group_size=2,
        num_minibatches=2,
        update_epochs=2,
        beta=0,
    )
    monkeypatch.setattr(grpo, "load_checkpoint", Mock(return_value=(model, {"params": params})))
    monkeypatch.setattr(grpo, "load_tokenizer", Mock(return_value=FakeTokenizer()))
    export = Mock()
    monkeypatch.setattr(grpo, "save_checkpoint", export)
    writer = MagicMock()
    monkeypatch.setattr(grpo, "SummaryWriter", Mock(return_value=writer))
    iterations: list[int] = []
    minibatch_sizes: list[int] = []
    now = [0.0]
    interrupt = [True]
    rollout_keys: list[jax.Array] = []

    def clock() -> float:
        return now[0]

    monkeypatch.setattr(grpo, "monotonic", clock)
    save = Mock(wraps=grpo.save_training_checkpoint)
    monkeypatch.setattr(grpo, "save_training_checkpoint", save)

    def collect(
        state: grpo.TrainState,
        model: GatedDeltaNet2LM,
        ref_params: Parameters,
        dataset: grpo.ProceduralDataset,
        tokenizer: grpo.Tokenizer,
        key: jax.Array,
        iteration: int,
        config: grpo.Config,
    ) -> tuple[grpo.GRPOBatch, dict[str, float], list[str], jax.Array]:
        assert ref_params == {}
        assert len(dataset) == config.total_updates
        if interrupt[0] and iteration == 3:
            raise RuntimeError("simulated interruption after unsaved work")
        now[0] = (599.0, 600.0, 1199.0, 1200.0)[iteration]
        if iteration == 2:
            assert int(state.step) == 8
            rollout_keys.append(jax.random.key_data(key))
        iterations.append(iteration)
        return make_batch(), {"charts/reward_mean": 0.5}, ["a completion"], jax.random.split(key)[0]

    def update(
        state: grpo.TrainState,
        model: GatedDeltaNet2LM,
        batch: grpo.GRPOBatch,
        *,
        temperature: float,
        clip_coef: float,
        beta: float,
    ) -> tuple[grpo.TrainState, grpo.Metrics]:
        minibatch_sizes.append(batch.prompts.shape[0])
        return state.replace(step=state.step + 1), {"losses/total": jnp.array(0.0, jnp.float32)}

    monkeypatch.setattr(grpo, "collect_rollout", collect)
    monkeypatch.setattr(grpo, "update", update)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        grpo.train(config)
    # No save at 599 seconds; save at 600; no second save at 1199.
    assert [call.args[3] for call in save.call_args_list] == [2]
    path = tmp_path / "test/checkpoint.msgpack"
    assert serialization.msgpack_restore(path.read_bytes())["iteration"] == 2
    export.assert_not_called()
    interrupt[0] = False
    resumed = grpo.train(config)
    assert int(resumed.step) == 16
    assert iterations == [0, 1, 2, 2, 3] and minibatch_sizes == [1] * 20
    np.testing.assert_array_equal(*rollout_keys)
    # Final save happens even though fewer than 600 seconds elapsed since restart.
    assert [call.args[3] for call in save.call_args_list] == [2, 4]
    assert serialization.msgpack_restore(path.read_bytes())["iteration"] == 4
    assert grpo.SummaryWriter.call_args.kwargs["purge_step"] == 5
    assert export.call_args.args[0] == tmp_path / "test/model-4"
    assert set(export.call_args.args[2]) == set(params)
    scalars = writer.__enter__.return_value.add_scalar.call_args_list
    assert {call.args[2] for call in scalars} == {8}
