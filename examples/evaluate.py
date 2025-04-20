"""Tasks."""

from __future__ import annotations
import json
from dataclasses import dataclass
from datasets import load_dataset
from pydantic import BaseModel
from pydantic_ai import Agent, models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_graph import End, Graph, GraphRunContext
from agentensor.module import AgentModule, ModuleState
from agentensor.tensor import TextTensor
from agentensor.train import Trainer


@dataclass
class GenerationTimeout(Evaluator[str, bool]):
    """The generation took too long."""

    threshold: float = 10.0

    async def evaluate(self, ctx: EvaluatorContext[str, bool]) -> bool:
        """Evaluate the time taken to generate the output."""
        return ctx.duration <= self.threshold  # pragma: no cover


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


@dataclass
class EvaluateState(ModuleState):
    """State of the graph."""

    agent_prompt: TextTensor = TextTensor(text="")


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


class AgentNode(AgentModule[EvaluateState, None, TextTensor]):
    """Agent node."""

    async def run(self, ctx: GraphRunContext[EvaluateState, None]) -> End[TextTensor]:  # type: ignore[override]
        """Run the agent node."""
        agent = Agent(
            model=model,
            system_prompt=ctx.state.agent_prompt.text,
            output_type=ClassificationResults,
        )
        assert ctx.state.input
        try:
            result = await agent.run(ctx.state.input.text)
            output = result.output
        except UnexpectedModelBehavior:
            output = "Error"  # type: ignore[assignment]

        output_tensor = TextTensor(
            str(output),
            parents=[ctx.state.input, ctx.state.agent_prompt],
            requires_grad=True,
        )

        return End(output_tensor)


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
    state = EvaluateState(
        agent_prompt=TextTensor(
            (
                "Classify the following text into one of the following "
                "categories: [expanding industry, new initiatives or programs, "
                "article publication, other]"
            ),
            requires_grad=True,
            model=model,
        )
    )
    graph = Graph(nodes=[AgentNode])
    trainer = Trainer(
        graph,
        state,
        AgentNode,  # type: ignore[arg-type]
        train_dataset=task.dataset["train"],
        test_dataset=task.dataset["test"],
    )
    trainer.test(limit_cases=10)
