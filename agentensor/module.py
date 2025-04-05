"""Module class."""

from pydantic_graph.nodes import BaseNode, DepsT, NodeRunEndT, StateT
from agentensor.tensor import TextTensor


class AgentModule(BaseNode[StateT, DepsT, NodeRunEndT]):
    """Agent module."""

    @classmethod
    def get_params(cls) -> list[TextTensor]:
        """Get the parameters of the module."""
        return [
            attr
            for attr in cls.__dict__.values()
            if isinstance(attr, TextTensor) and attr.requires_grad
        ]
