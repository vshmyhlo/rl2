"""Single-agent and multi-agent environment adapters for PPO."""

from rl2.envs.base import BaseEnv, EnvStep, StepInfo
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep

__all__ = ["AgentObservation", "BaseEnv", "BaseMultiAgentEnv", "EnvStep", "MultiAgentStep", "StepInfo"]
