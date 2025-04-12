"""Example usage of agentensor."""

from dataclasses import dataclass
from functools import partial
from typing import Any
import logfire
from pydantic_ai import Agent, models
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output, judge_output
from pydantic_graph import End, Graph, GraphRunContext
from agentensor.module import AgentModule
from agentensor.optim import Optimizer
from agentensor.tensor import TextTensor


logfire.configure(
    send_to_logfire="if-token-present",
    environment="development",
    service_name="evals",
)


@dataclass
class GraphState:
    """State class of the graph."""

    grad_agent: Agent


class AgentNode(AgentModule[GraphState, None, TextTensor]):
    """Agent node."""

    system_prompt = TextTensor("You are a helpful assistant.", requires_grad=True)

    def __init__(self, user_prompt: TextTensor):
        """Initialize the agent node.

        Args:
            user_prompt (TextTensor): The user prompt.
        """
        super().__init__()
        self.user_prompt = user_prompt

    async def run(self, ctx: GraphRunContext[GraphState]) -> End[TextTensor]:  # type: ignore[override]
        """Run the agent node."""
        agent = Agent(
            model="openai:gpt-4o-mini",
            system_prompt=self.system_prompt.text,
        )
        result = await agent.run(self.user_prompt.text)
        output = result.data

        output_tensor = TextTensor(output, requires_grad=True)
        output_tensor.parents.extend([self.user_prompt, self.system_prompt])
        output_tensor.grad_fn = ctx.state.grad_agent.run_sync

        return End(output_tensor)


async def run_graph(x: TextTensor, graph: Graph, state: GraphState) -> TextTensor:
    """Run the graph."""
    result = await graph.run(AgentNode(x), state=state)
    return result.output


@dataclass
class LLMTensorJudge(Evaluator[TextTensor, TextTensor, Any]):
    """LLM judge for text tensors.

    Adapted from pydantic_evals.evaluators.common.LLMJudge.
    """

    rubric: str
    model: models.Model | models.KnownModelName | None = None
    include_input: bool = True

    async def evaluate(
        self,
        ctx: EvaluatorContext[TextTensor, TextTensor, Any],
    ) -> EvaluationReason:
        """Evaluate the text tensor."""
        if self.include_input:
            grading_output = await judge_input_output(
                ctx.inputs.text, ctx.output.text, self.rubric, self.model
            )
        else:
            grading_output = await judge_output(
                ctx.output.text, self.rubric, self.model
            )
        return EvaluationReason(
            value=grading_output.pass_, reason=grading_output.reason
        )

    def build_serialization_arguments(self):
        """Build serialization arguments."""
        result = super().build_serialization_arguments()
        if (model := result.get("model")) and isinstance(model, models.Model):
            result["model"] = f"{model.system}:{model.model_name}"
        return result


@dataclass
class FormatJudge(LLMTensorJudge):
    """Alias for LLMTensorJudge."""

    pass


@dataclass
class ChineseLanguageJudge(LLMTensorJudge):
    """Alias for LLMTensorJudge."""

    pass


def main() -> None:
    """Main function."""
    grad_agent = Agent(
        model="openai:gpt-4o-mini", system_prompt="Answer the user's question."
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

    state = GraphState(grad_agent=grad_agent)
    nodes = [AgentNode(TextTensor("Hello, how are you?"))]
    graph = Graph(nodes=nodes)
    optimizer = Optimizer(nodes)  # type: ignore[arg-type]

    epochs = 15
    for i in range(epochs):
        partial_run_graph = partial(run_graph, graph=graph, state=state)
        report = dataset.evaluate_sync(partial_run_graph)
        report.print(include_input=True, include_output=True, include_durations=True)

        # Backward those failed cases
        for case in report.cases:
            losses = []
            if not case.assertions["ChineseLanguageJudge"].value:
                losses.append(case.assertions["ChineseLanguageJudge"].reason)
            if not case.assertions["FormatJudge"].value:
                losses.append(case.assertions["FormatJudge"].reason)
            if losses:
                case.output.backward(" ".join(losses))

        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {i + 1}")
        for param in optimizer.params:
            print(param.text)
        print()

        if report.averages().assertions >= 0.95:
            print("Optimization complete.")
            break


if __name__ == "__main__":
    main()
