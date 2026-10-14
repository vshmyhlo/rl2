from unittest.mock import Mock

import dialog
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gemma_helpers import tiny_gemma_model

from rl2 import train_grpo_reasoning as grpo
from rl2.gemma3 import Gemma3Carry, Gemma3LM, Parameters


def test_calculator_config_and_chat_format() -> None:
    config = grpo.load_config("configs/grpo_reasoning/gemma3.yaml")
    assert config.calculator_max_calls == 2 and grpo.Config().calculator_max_calls == 0
    for overrides in ({"calculator_max_calls": -1}, {"max_prompt_tokens": 32768}):
        with pytest.raises(ValueError):
            grpo.Config(**overrides)
    prompt = grpo.build_prompt("12 * 13?", config.prompt_examples, config.calculator_max_calls)
    content = prompt.removeprefix("<start_of_turn>user\n").removesuffix("<end_of_turn>\n<start_of_turn>model\n")
    assert prompt == dialog.Conversation(dialog.User(content)).as_text(format="gemma3")


class ToolTokenizer:
    def __init__(self) -> None:
        self.pieces = {
            0: "",
            1: "",
            2: "",
            3: "<calculator>",
            4: "12 * 13",
            5: "</calculator>",
            6: "\n<result>156</result>\n",
            7: "<answer>156</answer>",
            8: "",
            9: "1 / 0",
            10: "<answer>0</answer>",
        }

    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2

    def stop_ids(self) -> tuple[int, ...]:
        return (2, 8)

    def vocab_size(self) -> int:
        return 32

    def encode(self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool) -> list[int]:
        if add_bos:
            return [1, 4]
        assert not add_eos
        if text not in self.pieces.values():
            self.pieces[len(self.pieces)] = text
        return [next(token for token, piece in self.pieces.items() if piece == text)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[token] for token in ids)


def script_decoder(monkeypatch: pytest.MonkeyPatch, scripts: list[list[int]]) -> None:
    rows = len(scripts)
    carry = Gemma3Carry(cache={}, lengths=jnp.zeros(rows, jnp.int32), valid=jnp.zeros((rows, 1), jnp.bool_))
    logits = jnp.zeros((rows, 32), jnp.float32)  # (B, V)
    monkeypatch.setattr(grpo, "prefill_generation", Mock(return_value=(carry, logits)))
    streams = [iter(script) for script in scripts]

    def step(
        model: Gemma3LM,
        params: Parameters,
        carry: Gemma3Carry,
        logits: jax.Array,  # (B, V)
        forced: jax.Array,  # (B,)
        active: jax.Array,  # (B,)
        key: jax.Array,  # ()
        temperature: float,
        stop_token_ids: tuple[int, ...],
    ) -> grpo.DecodeStep:
        tokens = jnp.asarray(
            [
                int(force) if force >= 0 else next(stream) if enabled else 0
                for force, enabled, stream in zip(forced, active, streams, strict=True)
            ],
            dtype=jnp.int32,
        )  # (B,)
        return grpo.DecodeStep(
            carry=carry,
            logits=logits,
            tokens=tokens,
            log_probs=jnp.where(active & (forced < 0), -0.5, 0),
            key=jax.random.split(key)[0],
        )

    monkeypatch.setattr(grpo, "decode_step", step)


def test_calculator_rollout_and_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    script_decoder(
        monkeypatch,
        [
            [3, 4, 5, 7, 2],  # Call, result, answer, EOS.
            [10, 8],  # Wrong direct answer, end-of-turn, padding.
            [3, 9, 5, 3, 4, 5, 7, 2],  # Tool error followed by a successful retry.
            [3, 4, 5, 3, 4, 5, 3, 4, 5, 7, 2],  # Third call exceeds the limit.
        ],
    )
    config = grpo.Config(
        num_tasks=2,
        group_size=2,
        num_minibatches=1,
        max_prompt_tokens=3,
        max_new_tokens=14,
        calculator_max_calls=2,
        beta=0,
        duration_penalty_coef=0.1,
    )
    dataset = Mock()
    dataset.__getitem__ = Mock(return_value={"question": "What is 12 times 13?", "answer": "156"})

    def score(answer: str, entry: grpo.Entry) -> float:
        return float(answer == entry["answer"])

    dataset.score_answer = score
    tokenizer = ToolTokenizer()
    batch, diagnostics, samples, key = grpo.collect_rollout(
        Mock(params={}), Mock(cache_length=17), {}, dataset, tokenizer, jax.random.key(0), 0, config
    )
    assert batch.completions.shape == batch.mask.shape == batch.old_log_probs.shape == (4, 14)
    assert batch.prompts.shape == (4, 3) and batch.prompt_lengths.shape == (4,)
    assert batch.advantages.shape == (4,) and jax.random.key_data(key).shape == (2,)
    assert batch.mask.sum(axis=1).tolist() == [5, 2, 8, 11]
    assert grpo.transcript_mask(batch.mask).sum(axis=1).tolist() == [6, 2, 10, 14]
    assert not batch.mask[0, 3] and batch.completions[0, 3] == 6
    assert jnp.all(batch.old_log_probs[~batch.mask] == 0)
    transcripts = [tokenizer.decode(row.tolist()) for row in batch.completions]
    assert "<result>156</result>" in transcripts[0] and "<result>" not in transcripts[1]
    assert "Error: division by zero" in transcripts[2] and "<result>156</result>" in transcripts[2]
    assert "Error: calculator call limit reached" in transcripts[3]
    assert diagnostics["charts/task_reward_mean"] == 0.75
    assert diagnostics["charts/truncation_rate"] == 0
    assert diagnostics["charts/duration_penalty_mean"] == pytest.approx(0.1 * (5 + 2 + 8 + 11) / 4 / 14)
    assert "<result>156</result>" in samples and "You may use the calculator" in samples


@pytest.mark.parametrize("budget", [3, 4, 5], ids=["no-result-space", "no-continuation-space", "exact-fit"])
def test_calculator_budget(monkeypatch: pytest.MonkeyPatch, budget: int) -> None:
    script_decoder(monkeypatch, [[3, 4, 5, 7]])
    result = grpo.generate_with_calculator(
        Mock(cache_length=8),
        {},
        jnp.array([[1, 4, 0]], jnp.int32),
        jnp.array([2], jnp.int32),
        jax.random.key(0),
        tokenizer=ToolTokenizer(),
        group_size=1,
        max_new_tokens=budget,
        temperature=1.0,
        max_calls=1,
    )
    assert result.tokens.shape == result.mask.shape == (1, budget)
    assert result.mask.sum() == (4 if budget == 5 else 3)
    assert (6 in result.tokens[0]) == (budget == 5)


def test_incomplete_call_is_not_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    script_decoder(monkeypatch, [[3, 4, 2]])
    calculator = Mock(side_effect=AssertionError("incomplete calculator call"))
    monkeypatch.setattr(grpo, "calculate", calculator)
    result = grpo.generate_with_calculator(
        Mock(cache_length=8),
        {},
        jnp.array([[1, 4, 0]], jnp.int32),
        jnp.array([2], jnp.int32),
        jax.random.key(0),
        tokenizer=ToolTokenizer(),
        group_size=1,
        max_new_tokens=5,
        temperature=1.0,
        max_calls=1,
    )
    assert result.mask.tolist() == [[True, True, True, False, False]]
    calculator.assert_not_called()


def test_tool_tokens_condition_replay_but_have_no_loss() -> None:
    model, params = tiny_gemma_model()
    prompts = jnp.array([[2, 3, 0], [2, 4, 3]], jnp.int32)  # (B=2, P=3)
    lengths = jnp.array([2, 3], jnp.int32)  # (B,)
    carry, logits = grpo.prefill_generation(model, params, prompts, lengths, 1)  # logits: (B, V)
    key = jax.random.key(8)  # ()
    steps = []
    for forced in (-1, 6, -1):
        step = grpo.decode_step(
            model,
            params,
            carry,
            logits,
            jnp.full((2,), forced, jnp.int32),
            jnp.ones(2, jnp.bool_),
            key,
            0.7,
            (-1,),
        )
        carry, logits, key = step.carry, step.logits, step.key  # logits: (B, V); key: ()
        steps.append((step.tokens, step.log_probs))

    def stack(*values: jax.Array) -> jax.Array:  # (B,) each -> (B, C)
        return jnp.stack(values, axis=1)  # (B, C)

    tokens, old = jax.tree.map(stack, *steps)  # (B, C=3) each
    mask = jnp.array([[True, False, True]] * 2)  # (B, C)
    batch = grpo.GRPOBatch(
        prompts=prompts,
        prompt_lengths=lengths,
        completions=tokens,
        mask=mask,
        old_log_probs=old,
        ref_log_probs=old,
        advantages=jnp.array([1, -1], jnp.float32),
    )
    replay = grpo.completion_log_probs(model, params, batch, 0.7)  # (B, C)
    np.testing.assert_allclose(replay, old, atol=2e-6)
    assert jnp.all(replay[:, 1] == 0)

    def loss(log_probs: jax.Array) -> jax.Array:  # (B, C) -> ()
        return grpo.objective(log_probs, batch, 0.2, 0.1)[0]  # ()

    gradients = jax.grad(loss)(replay)  # (B, C)
    assert jnp.all(gradients[:, 1] == 0) and jnp.all(gradients[:, 2] != 0)
