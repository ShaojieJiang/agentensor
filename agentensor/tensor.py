"""Example module."""

from __future__ import annotations
from collections.abc import Callable


class TextTensor:
    """A tensor that represents a text."""

    def __init__(self, text: str, requires_grad: bool = False):
        """Initialize a TextTensor.

        Args:
            text (str): The text to represent.
            requires_grad (bool, optional): Whether to require gradients.
            Defaults to False.
        """
        self.text = text
        self.requires_grad = requires_grad
        self.text_grad = ""
        self.grad_fn: Callable | None = None
        self.parents: list[TextTensor] = []

    def backward(self, grad: str = "") -> None:
        """Backward pass for the TextTensor.

        Args:
            grad (str, optional): The gradient to backpropagate. Defaults to "".
        """
        if not grad:  # No gradient to backpropagate
            return

        self.text_grad = grad
        if self.grad_fn:
            for parent in self.parents:
                if not parent.requires_grad:
                    continue
                grad_to_parent = self.grad_fn(
                    f"Here is the input: \n\n>{parent.text}\n\nI got this "
                    f"output: \n\n>{self.text}\n\nHere is the feedback: \n\n"
                    f">{grad}\n\nHow should I improve the input to get a "
                    f"better output?"
                ).data
                parent.backward(grad_to_parent)

    def __str__(self) -> str:
        """Return the text as a string."""
        return self.text
