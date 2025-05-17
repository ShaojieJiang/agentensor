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
    model: models.Model | models.KnownModelName | str = "openai:gpt-4o"

    def get_params(self) -> list[TextTensor]:
        """Get the parameters of the module."""
        params = []
        for field_name in self.__dataclass_fields__.keys():
            field = getattr(self, field_name)
            if isinstance(field, TextTensor) and field.requires_grad:
                params.append(field)
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
