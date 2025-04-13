"""Tasks."""

from datasets import load_dataset
from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_graph import End, Graph, GraphRunContext
from agentensor.module import AgentModule, ModuleState
from agentensor.optim import Optimizer
from agentensor.tensor import TextTensor
from agentensor.train import Trainer


class HFMultiClassClassificationTask:
    """Multi-class classification task from Hugging Face."""

    def __init__(self, task_repo: str) -> None:
        """Initialize the multi-class classification task."""
        self.task_repo = task_repo
        self.dataset = self.prepare_dataset()

    def prepare_dataset(self) -> dict[str, Dataset]:
        """Return the Pydantic Evals dataset."""
        hf_dataset = load_dataset(self.task_repo, trust_remote_code=True)
        dataset = {}
        for split in hf_dataset.keys():
            cases = []
            for example in hf_dataset[split]:
                cases.append(
                    Case(
                        inputs={
                            "title": example["title"],
                            "content": example["content"],
                        },
                        expected_output=example["all_labels"],
                    )
                )
            dataset[split] = Dataset(cases=cases)
        return dataset


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


if __name__ == "__main__":
    task = HFMultiClassClassificationTask(
        task_repo="knowledgator/events_classification_biotech",
    )
    graph = Graph(nodes=[AgentNode])
    optimizer = Optimizer(graph)  # type: ignore[arg-type]
    trainer = Trainer(
        graph,
        AgentNode,  # type: ignore[arg-type]
        train_dataset=task.dataset["train"],
        eval_dataset=task.dataset["eval"],
    )
    report = trainer.evaluate("eval")
    report.print(include_input=True, include_output=True, include_durations=True)
