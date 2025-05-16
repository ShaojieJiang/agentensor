"""Module class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic_ai import Agent, models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from agentensor.tensor import TextTensor


@dataclass
class AgentModule(ABC):
    """Agent module."""

    system_prompt: TextTensor
    model: models.Model | models.KnownModelName | str

    def get_params(self) -> list[TextTensor]:
        """Get the parameters of the module."""
        params = []
        for _, attr in self.__dict__.items():
            if isinstance(attr, TextTensor) and attr.requires_grad:
                params.append(attr)
        return params

    async def __call__(self, state: dict) -> dict:
        """Run the agent node."""
        assert state["output"]
        agent = self.get_agent()
        try:
            result = await agent.run(state["output"].text)
            output = str(result.output)
        except UnexpectedModelBehavior:
            output = "Error"

        output_tensor = TextTensor(
            output,
            parents=[state["output"], self.system_prompt],
            requires_grad=True,
            model=self.model,
        )

        return {"output": output_tensor}

    @abstractmethod
    def get_agent(self) -> Agent:
        """Get agent instance."""
        pass  # programa: no cover
