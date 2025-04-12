"""Trainer."""

from typing import Any
from pydantic_evals import Dataset
from pydantic_graph import Graph
from agentensor.module import AgentModule, ModuleState
from agentensor.optim import Optimizer
from agentensor.tensor import TextTensor


class Trainer:
    """Trainer."""

    def __init__(
        self,
        graph: Graph[ModuleState, None, TextTensor],
        start_node: type[AgentModule],
        dataset: Dataset[TextTensor, TextTensor, Any],
        optimizer: Optimizer,
        epochs: int,
        stop_threshold: float = 0.95,
    ):
        """Initialize the trainer."""
        self.graph = graph
        self.start_node = start_node
        self.dataset = dataset
        self.optimizer = optimizer
        self.epochs = epochs
        self.stop_threshold = stop_threshold

    async def step(self, x: TextTensor) -> TextTensor:
        """Step the optimizer."""
        state = ModuleState(input=x)
        result = await self.graph.run(self.start_node(), state=state)  # type: ignore[arg-type]
        return result.output

    def train(self) -> None:
        """Train the model."""
        for i in range(self.epochs):
            report = self.dataset.evaluate_sync(self.step)
            report.print(
                include_input=True, include_output=True, include_durations=True
            )

            # Backward those failed cases
            for case in report.cases:
                losses = []
                for evaluator in case.assertions.values():
                    if not evaluator.value:
                        assert evaluator.reason
                        losses.append(evaluator.reason)
                if losses:
                    case.output.backward(" ".join(losses))

            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f"Epoch {i + 1}")
            for param in self.optimizer.params:
                print(param.text)  # pragma: no cover
            print()
            performance = report.averages().assertions
            assert performance is not None
            if performance >= self.stop_threshold:
                print("Optimization complete.")
                break
