"""Test module for the Module class."""

from unittest.mock import MagicMock, patch
import pytest
from agentensor.module import AgentModule, ModuleState
from agentensor.tensor import TextTensor


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    with patch("agentensor.tensor.Agent") as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        yield mock_agent


def test_module_state_initialization(mock_agent):
    """Test ModuleState initialization."""
    input_tensor = TextTensor("test input")
    state = ModuleState(input=input_tensor)

    assert isinstance(state.input, TextTensor)
    assert state.input.text == "test input"


def test_module_get_params(mock_agent):
    """Test AgentModule.get_params() method."""

    class TestModule(AgentModule):
        param1 = TextTensor("param1", requires_grad=True)
        param2 = TextTensor("param2", requires_grad=False)
        param3 = TextTensor("param3", requires_grad=True)
        non_param = "not a tensor"

        def __init__(self):
            pass

        def run(self, state: ModuleState) -> None:
            """Dummy run method for testing."""
            pass

    module = TestModule()
    params = module.get_params()

    assert len(params) == 2
    assert all(isinstance(p, TextTensor) for p in params)
    assert all(p.requires_grad for p in params)
    assert params[0].text == "param1"
    assert params[1].text == "param3"


def test_module_get_params_empty(mock_agent):
    """Test AgentModule.get_params() with no parameters."""

    class EmptyModule(AgentModule):
        non_param = "not a tensor"
        param = TextTensor("param", requires_grad=False)

        def __init__(self):
            pass

        def run(self, state: ModuleState) -> None:
            """Dummy run method for testing."""
            pass

    module = EmptyModule()
    params = module.get_params()

    assert len(params) == 0


def test_module_get_params_inheritance(mock_agent):
    """Test AgentModule.get_params() with inheritance."""

    class ParentModule(AgentModule):
        parent_param = TextTensor("parent", requires_grad=True)

        def __init__(self):
            pass

        def run(self, state: ModuleState) -> None:
            """Dummy run method for testing."""
            pass

    class ChildModule(ParentModule):
        child_param = TextTensor("child", requires_grad=True)

        def __init__(self):
            super().__init__()

        def run(self, state: ModuleState) -> None:
            """Dummy run method for testing."""
            pass

    module = ChildModule()
    params = module.get_params()

    assert len(params) == 2
    assert all(isinstance(p, TextTensor) for p in params)
    assert all(p.requires_grad for p in params)
    assert {p.text for p in params} == {"parent", "child"}
