"""Optimizer module."""

from pydantic_ai import Agent
from agentensor.module import AgentModule
from agentensor.tensor import TextTensor


class Optimizer:
    """Optimizer class."""

    def __init__(self, nodes: list[type[AgentModule]]):
        """Initialize the optimizer.

        Args:
            nodes (list[AgentModule]): The nodes to optimize.
        """
        self.params: list[TextTensor] = [
            param for node in nodes for param in node.get_params()
        ]
        self.agent = Agent(
            model="openai:gpt-4o-mini",
            system_prompt="Rewrite the system prompt given the feedback.",
        )

    def step(self) -> None:
        """Step the optimizer."""
        for param in self.params:
            if not param.text_grad:
                continue
            # TODO: make optimize function a parameter
            rewritten = self.agent.run_sync(
                f"Feedback: {param.text_grad}\nText: {param.text}"
            )
            param.text = rewritten.data

    def zero_grad(self) -> None:
        """Zero the gradients."""
        for param in self.params:
            param.text_grad = ""
