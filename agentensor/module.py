"""Module class."""

from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from agentensor.tensor import TextTensor


class AgentModule(BaseModel, ABC):
    """Agent module."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_prompt: TextTensor
    model: models.Model | models.KnownModelName | str = "openai:gpt-4o"

    def get_params(self) -> list[TextTensor]:
        """Get the parameters of the module."""
        params = []
        for field_name in self.__class__.model_fields.keys():
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
        except UnexpectedModelBehavior:  # pragma: no cover
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
        pass  # pragma: no cover
