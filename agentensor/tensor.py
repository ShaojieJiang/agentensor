"""Example module."""

from __future__ import annotations
from pydantic_ai import Agent, models


class TextTensor:
    """A tensor that represents a text."""

    def __init__(
        self,
        text: str,
        parents: list[TextTensor] | None = None,
        requires_grad: bool = False,
        model: models.Model | models.KnownModelName | str | None = None,
    ) -> None:
        """Initialize a TextTensor."""
        self.text = text
        self.requires_grad = requires_grad
        self.gradients = []
        self.agent = Agent(
            model=model or "openai:gpt-4o-mini",
            system_prompt="Answer the user's question.",
        )
        self.parents: list[TextTensor] = parents or []

    def backward(self, grad: str = "") -> None:
        """Backward pass for the TextTensor.

        Args:
            grad (str, optional): The gradient to backpropagate. Defaults to "".
        """
        if not grad:
            return

        if self.requires_grad:
            self.gradients.append(grad)
            for parent in self.parents:
                if not parent.requires_grad:
                    continue
                grad_to_parent = self.calc_grad(parent.text, self.text, grad)
                parent.backward(grad_to_parent)

    def calc_grad(self, input_text: str, output_text: str, grad: str) -> str:
        """Calculate the gradient for the TextTensor."""
        return self.agent.run_sync(
            f"Here is the input: \n\n>{input_text}\n\nI got this "
            f"output: \n\n>{output_text}\n\nHere is the feedback: \n\n"
            f">{grad}\n\nHow should I improve the input to get a "
            f"better output?"
        ).data

    @property
    def text_grad(self) -> str:
        """String representation of the gradients."""
        return " ".join(self.gradients)

    def zero_grad(self) -> None:
        """Zero the gradients."""
        self.gradients = []

    def __str__(self) -> str:
        """Return the text as a string."""
        return self.text
