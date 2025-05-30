"""Tasks."""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import TypedDict
from datasets import load_dataset
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from pydantic_ai import Agent, models
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from agentensor.module import AgentModule
from agentensor.tensor import TextTensor
from agentensor.train import Trainer


@dataclass
class GenerationTimeout(Evaluator[str, bool]):
    """The generation took too long."""

    threshold: float = 10.0

    async def evaluate(self, ctx: EvaluatorContext[str, bool]) -> EvaluationReason:
        """Evaluate the time taken to generate the output."""
        return EvaluationReason(
            value=ctx.duration <= self.threshold,
            reason=(
                f"The generation took {ctx.duration} seconds, which is longer "
                f"than the threshold of {self.threshold} seconds."
            ),
        )


@dataclass
class MultiLabelClassificationAccuracy(Evaluator):
    """Classification accuracy evaluator."""

    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        """Evaluate the accuracy of the classification."""
        try:
            output = json.loads(ctx.output.text)
        except json.JSONDecodeError:
            return False
        expected = ctx.expected_output
        return set(output) == set(expected)  # type: ignore[arg-type]


class EvaluateState(TypedDict):
    """State of the graph."""

    output: TextTensor


class ClassificationResults(BaseModel, use_attribute_docstrings=True):
    """Classification result for a data."""

    labels: list[str]
    """labels for this data point."""

    def __str__(self) -> str:
        """Return the string representation of the classification results."""
        return json.dumps(self.labels)


class HFMultiClassClassificationTask:
    """Multi-class classification task from Hugging Face."""

    def __init__(
        self,
        task_repo: str,
        evaluators: list[Evaluator],
        model: models.Model | models.KnownModelName | str | None = None,
    ) -> None:
        """Initialize the multi-class classification task."""
        self.task_repo = task_repo
        self.evaluators = evaluators
        self.model = model
        self.dataset = self._prepare_dataset()

    def _prepare_dataset(self) -> dict[str, Dataset]:
        """Return the Pydantic Evals dataset."""
        hf_dataset = load_dataset(self.task_repo, trust_remote_code=True)
        dataset = {}
        for split in hf_dataset.keys():
            cases = []
            for example in hf_dataset[split]:
                cases.append(
                    Case(
                        inputs=TextTensor(
                            f"Title: {example['title']}\nContent: {example['content']}",
                            model=self.model,
                        ),
                        expected_output=example["all_labels"],
                    )
                )
            dataset[split] = Dataset(cases=cases, evaluators=self.evaluators)
        return dataset


class AgentNode(AgentModule):
    """Agent node."""

    def get_agent(self) -> Agent:
        """Get agent instance."""
        return Agent(
            model=self.model or "openai:gpt-4o-mini",
            system_prompt=self.system_prompt.text,
            output_type=ClassificationResults,  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    model = OpenAIModel(
        model_name="llama3.2:1b",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
    )
    # model = "openai:gpt-4o-mini"

    task = HFMultiClassClassificationTask(
        task_repo="knowledgator/events_classification_biotech",
        evaluators=[GenerationTimeout(), MultiLabelClassificationAccuracy()],
        model=model,
    )
    graph = StateGraph(EvaluateState)
    graph.add_node(
        "agent",
        AgentNode(
            system_prompt=TextTensor(
                (
                    "Classify the following text into one of the following "
                    "categories: [expanding industry, new initiatives or programs, "
                    "article publication, other]"
                ),
                requires_grad=True,
                model=model,
            ),
            model=model,
        ),
    )
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    compiled_graph = graph.compile()
    trainer = Trainer(
        compiled_graph,
        train_dataset=task.dataset["train"],
        test_dataset=task.dataset["test"],
    )
    trainer.test(limit_cases=10)
