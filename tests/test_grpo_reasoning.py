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
    config = grpo.Config(
        num_tasks=1,
        group_size=2,
        num_minibatches=1,
        learning_rate=0.001,
        prompt_examples=[grpo.PromptExample("How many legs does a dog have?", "<answer>4</answer>")],
    )
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
    grpo.save_training_checkpoint(str(tmp_path), updated, key, 3, config, prompts_seen=3)
    data = (tmp_path / "checkpoint.msgpack").read_bytes()
    restored, restored_key, iteration, prompts_seen = grpo.restore_training_checkpoint(data, state)
    assert iteration == 3 and prompts_seen == 3
    for a, b in zip(jax.tree.leaves(updated), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(restored_key))
    # Restored optimizer moments produce the same next update.
    next_state, _ = grpo.update(updated, model, batch, temperature=1.0, clip_coef=0.2, beta=0.01)
    next_restored, _ = grpo.update(restored, model, batch, temperature=1.0, clip_coef=0.2, beta=0.01)
    for a, b in zip(jax.tree.leaves(next_state), jax.tree.leaves(next_restored), strict=True):
        np.testing.assert_array_equal(a, b)
    payload = serialization.msgpack_restore(data)
    payload["version"] = 999
    with pytest.raises(ValueError, match="version"):
        grpo.restore_training_checkpoint(serialization.msgpack_serialize(payload), state)
    save_checkpoint(tmp_path / "export", model, jax.device_get(updated.params), {})
    _, loaded = load_checkpoint(tmp_path / "export")
    for a, b in zip(jax.tree.leaves(updated.params), jax.tree.leaves(loaded["params"]), strict=True):
        np.testing.assert_array_equal(a, b)


def test_resume_uses_current_learning_rate_and_accepts_legacy_config(tmp_path: Path) -> None:
    config = grpo.Config(learning_rate=0.001)
    model = Mock()
    state = grpo.create_state(model, {"weight": jnp.array([1.0, 2.0], jnp.float32)}, config)
    grads = {"weight": jnp.array([0.2, -0.1], jnp.float32)}
    state = state.apply_gradients(grads=grads)
    key = jax.random.key(7)
    grpo.save_training_checkpoint(str(tmp_path), state, key, 12, config, prompts_seen=24)
    payload = serialization.msgpack_restore((tmp_path / "checkpoint.msgpack").read_bytes())
    # Older checkpoints stored only a subset of the configuration.
    payload["config"] = {"temperature": 1.0, "learning_rate": 0.001, "num_tasks": 2}
    payload.pop("prompts_seen")
    new_config = replace(config, learning_rate=0.002, temperature=0.5, num_tasks=4, group_size=8)
    template = grpo.create_state(model, {"weight": jnp.zeros(2, jnp.float32)}, new_config)
    restored, restored_key, iteration, prompts_seen = grpo.restore_training_checkpoint(
        serialization.msgpack_serialize(payload),
        template,
    )
    assert iteration == 12 and int(restored.step) == 1 and prompts_seen == 24
    # Explicit cumulative counts take precedence after task batch sizes have changed.
    payload["prompts_seen"] = 31
    assert grpo.restore_training_checkpoint(serialization.msgpack_serialize(payload), template)[3] == 31
    payload["prompts_seen"] = -1
    with pytest.raises(ValueError, match="prompts_seen"):
        grpo.restore_training_checkpoint(serialization.msgpack_serialize(payload), template)
    np.testing.assert_array_equal(jax.random.key_data(restored_key), jax.random.key_data(key))
    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(actual, expected)
    old_next = state.apply_gradients(grads=grads)
    new_next = restored.apply_gradients(grads=grads)
    # The saved Adam history is retained, but doubling the configured LR doubles the next update.
    np.testing.assert_allclose(
        new_next.params["weight"] - restored.params["weight"],
        2 * (old_next.params["weight"] - state.params["weight"]),
        atol=3e-7,
    )


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "extra_parameter", "optimizer_structure"])
def test_resume_rejects_incompatible_state(mismatch: str, tmp_path: Path) -> None:
    config = grpo.Config()
    state = grpo.create_state(Mock(), {"weight": jnp.ones(2, jnp.float32)}, config)
    grpo.save_training_checkpoint(str(tmp_path), state, jax.random.key(1), 0, config, prompts_seen=0)
    payload = serialization.msgpack_restore((tmp_path / "checkpoint.msgpack").read_bytes())
    if mismatch == "shape":
        payload["state"]["params"]["weight"] = np.ones(3, np.float32)
    elif mismatch == "dtype":
        payload["state"]["params"]["weight"] = np.ones(2, np.float16)
    elif mismatch == "extra_parameter":
        payload["state"]["params"]["extra"] = np.ones(2, np.float32)
    else:
        payload["state"]["opt_state"] = {}
    with pytest.raises(ValueError, match="model/optimizer.*incompatible"):
        grpo.restore_training_checkpoint(serialization.msgpack_serialize(payload), state)


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


def test_prompt_examples_precede_question_without_leaking_target_answer() -> None:
    examples = [grpo.PromptExample("How many legs do two dogs have?", "2 * 4 = 8.\n<answer>8</answer>")]
    tokenizer = Mock(wraps=FakeTokenizer())
    entry = {"question": "How many legs do three spiders have?", "answer": "24"}
    grpo.encode_prompts([entry], tokenizer, 3, examples)
    prompt = tokenizer.encode.call_args.args[0]
    assert "Question: How many legs do two dogs have?\n\nAnswer: 2 * 4 = 8.\n<answer>8</answer>" in prompt
    assert prompt.endswith("Question: How many legs do three spiders have?\n\nAnswer:")
    assert "24" not in prompt
    # Demonstration answers cannot earn reward for an empty generated completion.
    dataset = Mock()
    np.testing.assert_array_equal(grpo.score_completions(dataset, [entry], [""], 1), [0])
    dataset.score_answer.assert_not_called()


def test_format_group_samples_preserves_literal_multiline_text() -> None:
    prompt = "Question: first line\n\nsecond line\nAnswer:"
    completion = "thinking\n<answer>4</answer>\n```\n### Prompt\n```"
    sample = grpo.format_group_samples(prompt, [completion, ""], [0.5, 0.0])
    assert sample.count("### Shared prompt") == 1
    assert "### Shared prompt\n\n    Question: first line\n    \n    second line\n    Answer:" in sample
    assert "### Completion 1\n\n    thinking\n    <answer>4</answer>\n    ```\n    ### Prompt\n    ```" in sample
    assert "**Reward:** 0.500" in sample
    assert sample.endswith("### Completion 2\n\n    \n\n**Reward:** 0.000")
    assert sample.index("### Shared prompt") < sample.index("### Completion 1") < sample.index("### Completion 2")


def test_logged_completions_are_capped_and_stay_in_one_group(monkeypatch: pytest.MonkeyPatch) -> None:
    config = grpo.Config(group_size=8, max_prompt_tokens=3, max_new_tokens=1, beta=0)
    key = jax.random.key(1)
    dataset = Mock()
    dataset.__getitem__ = Mock(side_effect=[{"question": "First question"}, {"question": "Second question"}])
    tokenizer = Mock(wraps=FakeTokenizer())
    now = [0.0]
    decoded = [0]

    def clock() -> float:
        return now[0]

    def decode(ids: list[int]) -> str:
        now[0] += 0.25
        index = decoded[0]
        decoded[0] += 1
        return f"sample_{index}"

    original_score = grpo.score_completions

    def score(
        dataset: grpo.ProceduralDataset, entries: list[grpo.Entry], texts: list[str], group_size: int
    ) -> jax.Array:
        now[0] += 3.0
        return original_score(dataset, entries, texts, group_size)

    tokenizer.decode.side_effect = decode
    monkeypatch.setattr(grpo, "monotonic", clock)
    monkeypatch.setattr(grpo, "score_completions", score)
    ready = Mock(wraps=jax.block_until_ready)
    monkeypatch.setattr(grpo.jax, "block_until_ready", ready)
    monkeypatch.setattr(
        grpo,
        "generate",
        Mock(
            return_value=(
                jnp.full((16, 1), 3, jnp.int32),
                jnp.zeros((16, 1), jnp.float32),
                jnp.ones((16, 1), jnp.bool_),
                key,
            )
        ),
    )
    _, diagnostics, samples, _ = grpo.collect_rollout(Mock(params={}), Mock(), {}, dataset, tokenizer, key, 0, config)
    assert diagnostics["time/decoding_seconds"] == 4.0
    assert diagnostics["time/scoring_seconds"] == 3.0
    assert diagnostics["time/jax_seconds"] == 0.0
    ready.assert_not_called()
    assert samples.count("### Shared prompt") == 1
    assert "First question" in samples and "Second question" not in samples
    assert samples.count("### Completion ") == 4
    for index in range(4):
        assert f"### Completion {index + 1}\n\n    sample_{index}\n" in samples
    for index in range(4, 16):
        assert f"sample_{index}" not in samples


def test_collect_rollout_grouping_and_reference(
    tiny_model: tuple[GatedDeltaNet2LM, Parameters],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params = tiny_model
    config = grpo.Config(
        num_tasks=1,
        group_size=2,
        num_minibatches=1,
        max_prompt_tokens=3,
        max_new_tokens=3,
        prompt_examples=[grpo.PromptExample("How many legs does a dog have?", "<answer>4</answer>")],
    )
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
    assert samples.count("### Completion ") == 2
    encoded_prompt = tokenizer.encode.call_args.args[0]
    assert "### Shared prompt\n\n    " + encoded_prompt.replace("\n", "\n    ") in samples
    assert "### Completion 1\n\n    <answer>4</answer>\n\n**Reward:** 1.000" in samples
    assert samples.endswith("**Reward:** 0.000")
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
        ({"log_completion_count": 1}, "log_completion_count"),
        ({"log_completion_count": 5}, "log_completion_count"),
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
    assert config.group_size == 8 and config.total_updates == 100000
    assert config.task_config == {"min_animals": 1, "max_animals": 3, "min_instances": 1, "max_instances": 3}
    assert len(config.prompt_examples) == 2
    assert all(grpo.extract_answer(example.completion) for example in config.prompt_examples)
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


@pytest.mark.parametrize(
    "example,match",
    [
        ({"question": "q"}, "question and completion"),
        ({"question": "q", "completion": 8}, "question and completion"),
        ({"question": "q", "completion": "8"}, "<answer>"),
    ],
)
def test_invalid_prompt_examples(example: dict[str, Any], match: str, tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"prompt_examples": [example]}))
    with pytest.raises(ValueError, match=match):
        grpo.load_config(path)


def test_initialization_reports_stage_before_checkpoint_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(grpo.reasoning_gym, "create_dataset", Mock())
    monkeypatch.setattr(grpo, "load_tokenizer", Mock(return_value=FakeTokenizer()))
    monkeypatch.setattr(grpo, "load_checkpoint", Mock(side_effect=OSError("checkpoint unavailable")))
    with pytest.raises(OSError, match="checkpoint unavailable"):
        grpo.train(grpo.Config(checkpoint="missing-checkpoint"))
    output = capsys.readouterr().out
    assert "[init +" in output
    assert "Initializing JAX devices" in output
    assert "Loading pretrained checkpoint: missing-checkpoint (backend=jax, dtype=float32)" in output
    assert "Pretrained checkpoint loaded" not in output
    assert "Initialization complete" not in output


def test_training_loop_resume_and_logging_with_mocked_learning(
    tiny_model: tuple[GatedDeltaNet2LM, Parameters],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from unittest.mock import MagicMock

    model, params = tiny_model
    config = grpo.Config(
        log_dir=str(tmp_path),
        run_id="test",
        total_updates=8,
        log_interval=3,
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
    ready = Mock(wraps=jax.block_until_ready)
    monkeypatch.setattr(grpo.jax, "block_until_ready", ready)

    def collect(
        state: grpo.TrainState,
        model: GatedDeltaNet2LM,
        ref_params: Parameters,
        dataset: grpo.ProceduralDataset,
        tokenizer: grpo.Tokenizer,
        key: jax.Array,
        iteration: int,
        config: grpo.Config,
    ) -> tuple[grpo.GRPOBatch, dict[str, float], str, jax.Array]:
        assert ref_params == {}
        assert len(dataset) == config.total_updates * config.num_tasks
        if interrupt[0] and iteration == 5:
            raise RuntimeError("simulated interruption after unsaved work")
        now[0] = (599.0, 599.0, 599.0, 600.0, 1199.0, 1200.0, 1201.0, 1202.0)[iteration]
        if iteration == 4:
            assert int(state.step) == 16
            rollout_keys.append(jax.random.key_data(key))
        iterations.append(iteration)
        diagnostics = {
            "charts/reward_mean": 0.5,
            "time/generation_seconds": 1.0,
            "time/rollout_processing_seconds": 0.5,
            "time/jax_seconds": 1.5,
            "time/scoring_seconds": 0.125,
            "time/decoding_seconds": 0.25,
        }
        batch = grpo.GRPOBatch(*(jnp.repeat(x, config.num_tasks, axis=0) for x in make_batch()))
        return batch, diagnostics, "group preview", jax.random.split(key)[0]

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
    initial_stdout = capsys.readouterr().out
    assert "No training checkpoint found; starting from pretrained weights" in initial_stdout
    stages = [
        "Loading pretrained checkpoint:",
        "Pretrained checkpoint loaded:",
        "Transferring weights to device",
        "Model and optimizer ready",
        "Checking for saved training state:",
        "Writing run configuration:",
        "Opening TensorBoard writer:",
        "Initialization complete",
        "Collecting first rollout",
        "First rollout collected; starting optimization",
        "rollout_batches=1",
    ]
    positions = [initial_stdout.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert [line.split()[0] for line in initial_stdout.splitlines() if line.startswith("rollout_batches=")] == [
        "rollout_batches=1",
        "rollout_batches=2",
        "rollout_batches=3",
    ]
    # No save at 599 seconds; save at 600; no second save at 1199.
    assert [call.args[3] for call in save.call_args_list] == [4]
    path = tmp_path / "test/checkpoint.msgpack"
    assert serialization.msgpack_restore(path.read_bytes())["iteration"] == 4
    export.assert_not_called()
    interrupt[0] = False
    resumed = grpo.train(
        replace(
            config,
            log_completion_count=2,
            num_tasks=2,
            temperature=0.5,
            learning_rate=0.00001,
            task_config={"min_animals": 1, "max_animals": 3},
            prompt_examples=[grpo.PromptExample("How many legs do two dogs have?", "<answer>8</answer>")],
        )
    )
    resumed_stdout = capsys.readouterr().out
    assert "Training checkpoint read" in resumed_stdout
    assert "restoring model, optimizer, and RNG" in resumed_stdout
    assert "at rollout 4, optimizer step 16" in resumed_stdout
    assert "No training checkpoint found" not in resumed_stdout
    assert [line.split()[0] for line in resumed_stdout.splitlines() if line.startswith("rollout_batches=")] == [
        "rollout_batches=5",
        "rollout_batches=6",
        "rollout_batches=8",
    ]
    assert int(resumed.step) == 32
    assert iterations == [0, 1, 2, 3, 4, 4, 5, 6, 7] and minibatch_sizes == [1] * 20 + [2] * 16
    np.testing.assert_array_equal(*rollout_keys)
    # Final save happens even though fewer than 600 seconds elapsed since restart.
    assert [call.args[3] for call in save.call_args_list] == [4, 8]
    assert serialization.msgpack_restore(path.read_bytes())["iteration"] == 8
    assert grpo.SummaryWriter.call_args.kwargs["purge_step"] == 5
    assert export.call_args.args[0] == tmp_path / "test/model-8"
    assert set(export.call_args.args[2]) == set(params)
    scalars = writer.__enter__.return_value.add_scalar.call_args_list
    assert {call.args[2] for call in scalars} == {3, 8, 12}
    assert serialization.msgpack_restore(path.read_bytes())["prompts_seen"] == 12
    writer.__enter__.return_value.add_text.assert_any_call("samples/completions", "group preview", 12)

    # Only interval/final windows synchronize pending updates, including partial windows after resume.
    update_syncs = [
        call
        for call in ready.call_args_list
        if isinstance(call.args[0], tuple) and isinstance(call.args[0][0], grpo.TrainState)
    ]
    assert len(update_syncs) == 3
    assert [call.args[1] for call in scalars if call.args[0] == "time/window_rollouts"] == [3, 2, 2]
    assert [call.args[1] for call in scalars if call.args[0] == "time/jax_seconds"] == [4.5, 3.0, 3.0]
    assert [call.args[1] for call in scalars if call.args[0] == "time/scoring_seconds"] == [0.375, 0.25, 0.25]
    assert "window_rollouts=3 jax=4.500s scoring=0.375s" in initial_stdout
    assert "window_rollouts=2 jax=3.000s scoring=0.250s" in resumed_stdout
    assert " loss=" not in next(line for line in initial_stdout.splitlines() if line.startswith("rollout_batches=1 "))
