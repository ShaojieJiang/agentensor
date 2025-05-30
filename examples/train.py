"""Example usage of agentensor."""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph
from pydantic_ai import Agent, models
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from agentensor.loss import LLMTensorJudge
from agentensor.module import AgentModule
from agentensor.optim import Optimizer
from agentensor.tensor import TextTensor
from agentensor.train import Trainer


@dataclass
class ChineseLanguageJudge(LLMTensorJudge):
    """Chinese language judge."""

    rubric: str = "The output should be in Chinese."
    model: models.Model | models.KnownModelName = "openai:gpt-4o-mini"
    include_input = True


@dataclass
class FormatJudge(LLMTensorJudge):
    """Format judge."""

    rubric: str = "The output should start by introducing itself."
    model: models.Model | models.KnownModelName = "openai:gpt-4o-mini"
    include_input = True


class TrainState(TypedDict):
    """State of the graph."""

    output: TextTensor


class AgentNode(AgentModule):
    """Agent node."""

    def get_agent(self) -> Agent:
        """Get agent instance."""
        return Agent(
            model=self.model or "openai:gpt-4o-mini",
            system_prompt=self.system_prompt.text,
        )


def main() -> None:
    """Main function."""
    if os.environ.get("LOGFIRE_TOKEN", None):
        import logfire

        logfire.configure(
            send_to_logfire="if-token-present",
            environment="development",
            service_name="evals",
        )
    model = OpenAIModel(
        model_name="llama3.2:1b",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
    )
    # model="openai:gpt-4o-mini"

    dataset = Dataset[TextTensor, TextTensor, Any](
        cases=[
            Case(
                inputs=TextTensor("Hello, how are you?", model=model),
                metadata={"language": "English"},
            ),
            Case(
                inputs=TextTensor("こんにちは、元気ですか？", model=model),
                metadata={"language": "Japanese"},
            ),
        ],
        evaluators=[
            ChineseLanguageJudge(model=model),
            FormatJudge(model=model),
        ],
    )

    graph = StateGraph(TrainState)
    graph.add_node(
        "agent",
        AgentNode(
            system_prompt=TextTensor(
                "You are a helpful assistant.", requires_grad=True, model=model
            ),
            model=model,
        ),
    )
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    compiled_graph = graph.compile()
    optimizer = Optimizer(graph, model=model)
    trainer = Trainer(
        compiled_graph,
        train_dataset=dataset,
        optimizer=optimizer,
        epochs=15,
    )
    trainer.train()


if __name__ == "__main__":
    main()
