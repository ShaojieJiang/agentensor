"""Example usage of agentensor."""

import os
from dataclasses import dataclass
from typing import Any
from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_graph import End, Graph, GraphRunContext
from agentensor.loss import LLMTensorJudge
from agentensor.module import AgentModule, ModuleState
from agentensor.optim import Optimizer
from agentensor.tensor import TextTensor
from agentensor.train import Trainer


@dataclass
class FormatJudge(LLMTensorJudge):
    """Alias for LLMTensorJudge."""

    pass


@dataclass
class ChineseLanguageJudge(LLMTensorJudge):
    """Alias for LLMTensorJudge."""

    pass


class AgentNode(AgentModule[ModuleState, None, TextTensor]):
    """Agent node."""

    system_prompt = TextTensor("You are a helpful assistant.", requires_grad=True)

    async def run(self, ctx: GraphRunContext[ModuleState, None]) -> End[TextTensor]:  # type: ignore[override]
        """Run the agent node."""
        agent = Agent(
            model="openai:gpt-4o-mini",
            system_prompt=self.system_prompt.text,
        )
        result = await agent.run(ctx.state.input.text)
        output = result.data

        output_tensor = TextTensor(
            output, parents=[ctx.state.input, self.system_prompt], requires_grad=True
        )

        return End(output_tensor)


def main() -> None:
    """Main function."""
    if os.environ.get("LOGFIRE_TOKEN", None):
        import logfire

        logfire.configure(
            send_to_logfire="if-token-present",
            environment="development",
            service_name="evals",
        )

    dataset = Dataset[TextTensor, TextTensor, Any](
        cases=[
            Case(
                inputs=TextTensor("Hello, how are you?"),
                metadata={"language": "English"},
            ),
            Case(
                inputs=TextTensor("こんにちは、元気ですか？"),
                metadata={"language": "Japanese"},
            ),
        ],
        evaluators=[
            ChineseLanguageJudge(
                rubric="The output should be in Chinese.",
                model="openai:gpt-4o-mini",
                include_input=True,
            ),
            FormatJudge(
                rubric="The output should start by introducing itself.",
                model="openai:gpt-4o-mini",
                include_input=True,
            ),
        ],
    )

    graph = Graph(nodes=[AgentNode])
    optimizer = Optimizer(graph)  # type: ignore[arg-type]
    trainer = Trainer(graph, AgentNode, dataset, optimizer, 15)  # type: ignore[arg-type]
    trainer.train()


if __name__ == "__main__":
    main()
