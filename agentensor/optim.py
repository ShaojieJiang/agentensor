"""Optimizer module."""

from typing import Callable
from agentensor.module import AgentModule
from agentensor.tensor import TextTensor


class Optimizer:
    """Optimizer class."""

    def __init__(
        self, nodes: list[type[AgentModule]], optimize_fn: Callable[[str, str], str]
    ):
        """Initialize the optimizer.

        Args:
            nodes (list[AgentModule]): The nodes to optimize.
        """
        self.params: list[TextTensor] = [
            param for node in nodes for param in node.get_params()
        ]
        self.optimize_fn: Callable[[str, str], str] = optimize_fn

    def step(self) -> None:
        """Step the optimizer."""
        for param in self.params:
            if not param.text_grad:
                continue
            param.text = self.optimize_fn(param.text, param.text_grad)

    def zero_grad(self) -> None:
        """Zero the gradients."""
        for param in self.params:
            param.text_grad = ""
