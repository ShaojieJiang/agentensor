"""Example usage of agentensor."""

from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_graph import End, Graph, GraphRunContext
from agentensor.module import AgentModule
from agentensor.optim import Optimizer
from agentensor.tensor import TextTensor


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


loss_agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="Evaluate whether the text is in Japanese.",
)
grad_agent = Agent(
    model="openai:gpt-4o-mini", system_prompt="Answer the user's question."
)

state = GraphState(grad_agent=grad_agent)
nodes = [AgentNode]
graph = Graph(nodes=nodes)
optimizer = Optimizer(nodes)  # type: ignore[arg-type]
x = TextTensor("Hello, how are you?")

epochs = 15
for i in range(epochs):
    result = graph.run_sync(AgentNode(x), state=state)

    eval = loss_agent.run_sync(result.output.text)
    result.output.backward(eval.data)

    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {i + 1}")
    print(result.output.text)
    for param in optimizer.params:
        print(param.text)
    print()
