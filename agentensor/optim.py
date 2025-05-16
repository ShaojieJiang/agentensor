"""Optimizer module."""

from pydantic_ai import Agent, models
from agentensor.module import ModuleState
from agentensor.tensor import TextTensor


class Optimizer:
    """Optimizer class."""

    def __init__(
        self,
        state: ModuleState,
        model: models.Model | models.KnownModelName | str | None = None,
    ) -> None:
        """Initialize the optimizer."""
        self.params: list[TextTensor] = [
            attr
            for attr in vars(state).values()
            if isinstance(attr, TextTensor) and attr.requires_grad
        ]
        self.agent: Agent = Agent(
            model=model or "openai:gpt-4o-mini",
            system_prompt="Rewrite the system prompt given the feedback.",
        )

    def step(self) -> None:
        """Step the optimizer."""
        for param in self.params:
            if not param.text_grad:
                continue
            param.text = self.optimize(param.text, param.text_grad)

    def zero_grad(self) -> None:
        """Zero the gradients."""
        for param in self.params:
            param.zero_grad()

    def optimize(self, text: str, grad: str) -> str:
        """Optimize the text."""
        return self.agent.run_sync(f"Feedback: {grad}\nText: {text}").data
