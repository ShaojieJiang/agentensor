"""Module class."""

from dataclasses import dataclass
from agentensor.tensor import TextTensor


@dataclass
class ModuleState:
    """State of the graph."""

    input: TextTensor | None = None


class AgentModule:
    """Agent module."""

    def get_params(self) -> list[TextTensor]:
        """Get the parameters of the module."""
        params = []
        for _, attr in self.__dict__.items():
            if isinstance(attr, TextTensor) and attr.requires_grad:
                params.append(attr)
        return params
