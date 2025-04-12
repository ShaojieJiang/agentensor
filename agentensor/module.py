"""Module class."""

from dataclasses import dataclass
from pydantic_graph.nodes import BaseNode, DepsT, NodeRunEndT, StateT
from agentensor.tensor import TextTensor


@dataclass
class ModuleState:
    """State of the graph."""

    input: TextTensor


class AgentModule(BaseNode[StateT, DepsT, NodeRunEndT]):
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
