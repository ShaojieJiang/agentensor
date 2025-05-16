"""Module class."""

from dataclasses import dataclass
from agentensor.tensor import TextTensor


@dataclass
class ModuleState:
    """State of the graph."""

    input: TextTensor | None = None


class AgentModule:
    """Agent module."""

    @classmethod
    def get_params(cls) -> list[TextTensor]:
        """Get the parameters of the module."""
        params = []
        for base in cls.__mro__:
            for _, attr in base.__dict__.items():
                if isinstance(attr, TextTensor) and attr.requires_grad:
                    params.append(attr)
        return params
