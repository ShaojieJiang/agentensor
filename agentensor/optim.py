"""Optimizer module."""

from pydantic_ai import Agent
from agentensor.module import AgentModule
from agentensor.tensor import TextTensor


class Optimizer:
    """Optimizer class."""

    def __init__(self, nodes: list[type[AgentModule]], agent: Agent):
        """Initialize the optimizer."""
        self.params: list[TextTensor] = [
            param for node in nodes for param in node.get_params()
        ]
        self.agent: Agent = agent

    def step(self) -> None:
        """Step the optimizer."""
        for param in self.params:
            if not param.text_grad:
                continue
            param.text = self.optimize(param.text, param.text_grad)

    def zero_grad(self) -> None:
        """Zero the gradients."""
        for param in self.params:
            param.text_grad = ""

    def optimize(self, text: str, grad: str) -> str:
        """Optimize the text."""
        return self.agent.run_sync(f"Feedback: {grad}\nText: {text}").data
