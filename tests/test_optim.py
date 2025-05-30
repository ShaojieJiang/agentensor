"""Test module for the Optimizer class."""

from unittest.mock import MagicMock, patch
import pytest
from agentensor.module import AgentModule
from agentensor.optim import Optimizer
from agentensor.tensor import TextTensor


@pytest.fixture
def mock_graph():
    """Create a mock graph for testing."""
    mock_graph = MagicMock()
    return mock_graph


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    with patch("agentensor.optim.Agent") as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        yield mock_agent


@pytest.fixture
def mock_module_class():
    """Create a mock module class for testing."""

    class MockModule(AgentModule):
        system_prompt: TextTensor = TextTensor("system", requires_grad=True)
        param1: TextTensor = TextTensor("initial text 1", requires_grad=True)
        param2: TextTensor = TextTensor("initial text 2", requires_grad=True)

        def run(self, state):
            pass

        def get_agent(self):
            """Dummy get_agent method for testing."""
            pass

    return MockModule


def test_optimizer_initialization(mock_graph, mock_agent):
    """Test Optimizer initialization."""
    optimizer = Optimizer(mock_graph)
    assert optimizer.agent is not None
    assert isinstance(optimizer.params, list)


def test_optimizer_zero_grad(mock_graph, mock_agent):
    """Test zero_grad method."""
    optimizer = Optimizer(mock_graph)
    optimizer.params = [
        TextTensor("text1", requires_grad=True),
        TextTensor("text2", requires_grad=True),
    ]

    # Set some gradients
    optimizer.params[0].gradients = ["grad1"]
    optimizer.params[1].gradients = ["grad2"]

    optimizer.zero_grad()

    assert optimizer.params[0].text_grad == ""
    assert optimizer.params[1].text_grad == ""


def test_optimizer_step(mock_graph, mock_agent):
    """Test step method."""
    optimizer = Optimizer(mock_graph)
    optimizer.params = [
        TextTensor("text1", requires_grad=True),
        TextTensor("text2", requires_grad=True),
    ]

    # Set some gradients
    optimizer.params[0].gradients = ["grad1"]
    optimizer.params[1].gradients = ["grad2"]

    # Mock the agent's response
    mock_agent.run_sync.return_value.data = "optimized text"

    optimizer.step()

    # Verify the agent was called for each parameter with gradient
    assert mock_agent.run_sync.call_count == 2
    assert optimizer.params[0].text == "optimized text"
    assert optimizer.params[1].text == "optimized text"


def test_optimizer_step_no_grad(mock_graph, mock_agent):
    """Test step method when there are no gradients."""
    optimizer = Optimizer(mock_graph)
    optimizer.params = [
        TextTensor("text1", requires_grad=True),
        TextTensor("text2", requires_grad=True),
    ]

    # No gradients set
    optimizer.step()

    # Verify the agent was not called
    assert mock_agent.run_sync.call_count == 0
    assert optimizer.params[0].text == "text1"
    assert optimizer.params[1].text == "text2"
