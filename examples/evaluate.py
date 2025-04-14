"""Tasks."""

from __future__ import annotations
from dataclasses import dataclass
from datasets import load_dataset
from pydantic_ai import Agent, models
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_graph import End, Graph, GraphRunContext
from agentensor.module import AgentModule, ModuleState
from agentensor.tensor import TextTensor
from agentensor.train import Trainer


@dataclass
class UnstoppableGibberish(Evaluator[str, bool]):
    """The main metric for unstoppable gibberish is the generation taking too long."""

    threshold: float = 10.0

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> bool:
        """Evaluate the time taken to generate the output."""
        return ctx.duration <= self.threshold  # pragma: no cover


class HFMultiClassClassificationTask:
    """Multi-class classification task from Hugging Face."""

    def __init__(
        self,
        task_repo: str,
        evaluators: list[Evaluator[str, bool]],
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


class AgentNode(AgentModule[ModuleState, None, TextTensor]):
    """Agent node."""

    async def run(self, ctx: GraphRunContext[ModuleState, None]) -> End[TextTensor]:  # type: ignore[override]
        """Run the agent node."""
        agent = Agent(
            model=model,
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
class EvaluateState(ModuleState):
    """State of the graph."""

    agent_prompt: TextTensor = TextTensor(text="")


if __name__ == "__main__":
    model = OpenAIModel(
        model_name="llama3.2:1b",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
    )

    task = HFMultiClassClassificationTask(
        task_repo="knowledgator/events_classification_biotech",
        evaluators=[UnstoppableGibberish()],
        model=model,
    )
    state = EvaluateState(
        agent_prompt=TextTensor(
            "You are a helpful assistant.",
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
    print(len(trainer.test_dataset.cases))
