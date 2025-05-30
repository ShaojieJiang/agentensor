"""Test module for the Module class."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from agentensor.module import AgentModule
from agentensor.tensor import TextTensor


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    with patch("agentensor.tensor.Agent") as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        yield mock_agent


def test_module_get_params():
    """Test AgentModule.get_params() method."""

    class TestModule(AgentModule):
        system_prompt: TextTensor = TextTensor("param1", requires_grad=True)
        param2: TextTensor = TextTensor("param2", requires_grad=False)
        param3: TextTensor = TextTensor("param3", requires_grad=True)
        model: str = "openai:gpt-4o"
        non_param: str = "not a tensor"

        def get_agent(self):
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
        system_prompt: TextTensor = TextTensor("param", requires_grad=False)
        non_param: str = "not a tensor"

        def get_agent(self):
            """Dummy run method for testing."""
            pass

    module = EmptyModule()
    params = module.get_params()

    assert len(params) == 0


def test_module_get_params_inheritance():
    """Test AgentModule.get_params() with inheritance."""

    class ParentModule(AgentModule):
        system_prompt: TextTensor = TextTensor("parent", requires_grad=True)

        def get_agent(self):
            """Dummy run method for testing."""
            pass

    class ChildModule(ParentModule):
        child_param: TextTensor = TextTensor("child", requires_grad=True)

        def get_agent(self):
            """Dummy run method for testing."""
            pass

    module = ChildModule()
    params = module.get_params()

    assert len(params) == 2
    assert all(isinstance(p, TextTensor) for p in params)
    assert all(p.requires_grad for p in params)
    assert {p.text for p in params} == {"parent", "child"}


@pytest.mark.asyncio
async def test_module_call():
    class TestModule(AgentModule):
        system_prompt: TextTensor = TextTensor("system prompt", requires_grad=True)

        def get_agent(self):
            """Dummy run method for testing."""
            mock_agent = AsyncMock()
            run_output = MagicMock()
            run_output.output = "Output text"
            mock_agent.run.return_value = run_output
            return mock_agent

    module = TestModule()

    result = await module({"output": TextTensor("Input text")})
    assert isinstance(result["output"], TextTensor)
    assert result["output"].text == "Output text"
