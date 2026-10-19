# Environments for agentic GRPO

Start with Prime Intellect's Environments Hub and Verifiers. The project already includes `verifiers`, and its environments support tool execution, multi-turn rollouts, and rewards.

| Collection | What you get | Why consider it |
|---|---|---|
| [Prime Environments Hub](https://app.primeintellect.ai/dashboard/environments) + [Verifiers](https://docs.primeintellect.ai/verifiers/v1/overview) | Installable environments, tool execution, stateful multi-turn rollouts, reward functions, and environment mixtures. | First choice for plugging existing tasks into the actor. |
| [OpenEnv](https://github.com/huggingface/OpenEnv) | Containerized environments with `reset`/`step` interfaces; coding, text games, browser tasks, and other domains. | Useful for separating environment servers from model inference. |
| [AgentGym](https://github.com/WooooDyy/AgentGym) + [AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL) | WebShop, ALFWorld, ScienceWorld, TextCraft, SQL, and tool-use environments. The RL framework explicitly supports GRPO. | Broad collection for multi-step decision-making experiments. |
| [AppWorld](https://github.com/StonyBrookNLP/appworld) | Stateful simulated apps with executable APIs, task evaluation, and explicit train/dev/test splits. | Realistic workflows involving several tools and persistent state. |
| [τ²-bench](https://github.com/sierra-research/tau2-bench) | Customer-service conversations, tools, policies, and simulated users across retail, airline, and telecom domains. | Multi-turn interaction with users and tools; budget for user-simulator inference. |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym) | A common interface to MiniWoB++, WebArena, WorkArena, and other browser benchmarks. | Browser agents; requires more environment setup than simple Python tools. |

## Starting point

The prototype uses `verifiers==0.4.0` and its `verifiers.v1` API. `AdditionTaskset.load()` supplies five local addition tasks; it and the MCP `add` tool live in `addition_env.py`. The actor uses the documented `vf.make_agent(vf.AgentConfig(...))` factory and `agent.run(task)` interface. The actor owns vLLM and runs Verifiers' `null` chat harness in local subprocesses for up to three model turns. An exact final integer answer earns 1, otherwise 0.

The actor samples four independent rollouts of the same task concurrently (`group_size=4` on `Actor`, minimum 2), then scores the completed group. Each rollout receives `(reward - group_mean) / (group_std + 1e-4)`, using population standard deviation. Equal-reward groups have zero advantages. The batcher receives a compact group: shared task ID/prompt plus each rollout's training sequence, reward, advantage, final answer, turn/token counts, duration, and stop condition. Full Verifiers traces are discarded after scoring; generation or scoring failures prevent submission.

Run with `uv run --extra bfcl python -m rl2.train_grpo_v2` on a host with two visible GPUs. Ray reserves one GPU for `Actor` (inherited by its vLLM subprocess) and the other for `Learner`; the batcher runs on CPU. The first rollout provisions the harness's Python dependencies using uv. No external dataset or paid inference service is needed.

Each `Rollout.sequence` contains exactly one complete multi-turn token sequence: exact token IDs, aligned sampling log-probabilities, and a loss mask excluding prompts and tool results. Missing tokens/log-probabilities, inconsistent lengths, and multiple branches fail the group before submission. `TrainClientConfig` renders Qwen3 prompts and parses tool calls locally, using vLLM 0.27.0's native `/inference/v1/generate` endpoint to obtain sampled tokens and log-probabilities. This endpoint is registered by the existing server entrypoint; no additional server or launch flags are needed. Batching, optimization, and weight publishing remain placeholders.

For Qwen3-0.6B, defer AppWorld and long browser workflows until basic tool calling works reliably. This is a practical recommendation, not a measured comparison of this model on these suites.

For GRPO, select environments that can reset the same task into independent copies and produce multiple trajectories with meaningful reward differences. Check rollout cost, termination rules, and available training splits.

Verifiers owns the conversation/tool loop inside `Actor.collect_group()`, while retaining the Ray actors, batcher, and learner. The [Verifiers v1 overview](https://docs.primeintellect.ai/verifiers/v1/overview#trace) describes traces and token/log-probability capture with prime-rl; the [prime-rl training guide](https://docs.primeintellect.ai/prime-rl/training) covers training runs.

Checked 2026-10-19: the lockfile and installed package use `0.4.0`, the [latest stable PyPI release](https://pypi.org/project/verifiers/); `0.4.1.dev41` is a prerelease. The [standalone-agent](https://docs.primeintellect.ai/verifiers/v1/agent) and [taskset](https://docs.primeintellect.ai/verifiers/v1/tasksets) interfaces above match the current docs. The local subprocess runtime is for prototyping; the [architecture guide](https://docs.primeintellect.ai/verifiers/v1/architecture) recommends isolated runtimes for production training.
