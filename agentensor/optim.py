"""Optimizer module."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from agentensor.module import AgentModule
from agentensor.tensor import TextTensor


class Optimizer:
    """Optimizer class."""

    def __init__(
        self,
        graph: StateGraph,
        model: str | BaseChatModel = "gpt-4o-mini",
    ) -> None:
        """Initialize the optimizer."""
        self.params: list[TextTensor] = [
            param
            for node in graph.nodes.values()
            if isinstance(node.runnable.afunc, AgentModule)  # type: ignore[attr-defined]
            for param in node.runnable.afunc.get_params()  # type: ignore[attr-defined]
        ]
        if isinstance(model, str):
            self.model = init_chat_model(model)
        else:  # pragma: no cover
            self.model = model

    def step(self) -> None:
        """Step the optimizer."""
        for param in self.params:
            if not param.text_grad:
                continue
            param.text = self.optimize(param.text, param.text_grad)

    def zero_grad(self) -> None:
        """Zero the gradients."""
        for param in self.params:
            param.zero_grad()

    def optimize(self, text: str, grad: str) -> str:
        """Optimize the text."""
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=f"Feedback: {grad}\nText: {text}")]}
        )
        return result["messages"][-1].content

    @property
    def agent(self) -> CompiledGraph:
        """Get the agent."""
        return create_react_agent(
            self.model, tools=[], prompt="Rewrite the system prompt given the feedback."
        )
