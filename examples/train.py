"""Example usage of agentensor."""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any
from pydantic_ai import Agent, models
from pydantic_evals import Case, Dataset
from pydantic_graph import End, Graph, GraphRunContext
from agentensor.loss import LLMTensorJudge
from agentensor.module import AgentModule, ModuleState
from agentensor.optim import Optimizer
from agentensor.tensor import TextTensor
from agentensor.train import Trainer


@dataclass
class ChineseLanguageJudge(LLMTensorJudge):
    """Chinese language judge."""

    rubric: str = "The output should be in Chinese."
    model: models.KnownModelName = "openai:gpt-4o-mini"
    include_input = True


@dataclass
class FormatJudge(LLMTensorJudge):
    """Format judge."""

    rubric: str = "The output should start by introducing itself."
    model: models.KnownModelName = "openai:gpt-4o-mini"
    include_input = True


class AgentNode(AgentModule[ModuleState, None, TextTensor]):
    """Agent node."""

    async def run(self, ctx: GraphRunContext[ModuleState, None]) -> End[TextTensor]:  # type: ignore[override]
        """Run the agent node."""
        agent = Agent(
            model="openai:gpt-4o-mini",
            system_prompt=ctx.state.agent_prompt.text,
        )
        result = await agent.run(ctx.state.input.text)
        output = result.data

        output_tensor = TextTensor(
            output,
            parents=[ctx.state.input, ctx.state.agent_prompt],
            requires_grad=True,
        )

        return End(output_tensor)


@dataclass
class TrainState(ModuleState):
    """State of the graph."""

    agent_prompt: TextTensor | None = None


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
            ChineseLanguageJudge(),
            FormatJudge(),
        ],
    )

    state = TrainState(
        agent_prompt=TextTensor("You are a helpful assistant.", requires_grad=True)
    )
    graph = Graph(nodes=[AgentNode])
    optimizer = Optimizer(state)  # type: ignore[arg-type]
    trainer = Trainer(
        graph,
        state,
        AgentNode,  # type: ignore[arg-type]
        train_dataset=dataset,
        optimizer=optimizer,
        epochs=15,
    )
    trainer.train()


if __name__ == "__main__":
    main()
