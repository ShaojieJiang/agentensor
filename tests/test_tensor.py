"""Test module for the TextTensor class."""

from unittest.mock import MagicMock, patch
import pytest
from agentensor.tensor import TextTensor


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    with patch("agentensor.tensor.Agent") as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        yield mock_agent


def test_text_tensor_requires_grad_false(mock_agent):
    """Test TextTensor behavior when requires_grad is False."""
    # Create a tensor with requires_grad=False
    tensor = TextTensor("test text", requires_grad=False)

    # Verify initial state
    assert tensor.text == "test text"
    assert tensor.requires_grad is False
    assert tensor.text_grad == ""
    assert tensor.parents == []

    # Try to perform backward pass
    tensor.backward("some gradient")

    # Verify that the gradient was not set and no backward pass was performed
    assert tensor.text_grad == ""
    # Verify that the agent was not called
    assert not mock_agent.run_sync.called


def test_backward_without_grad_and_requires_grad_false():
    """Test backward pass when requires_grad is False and grad is empty string."""
    # Create a TextTensor with requires_grad=False
    tensor = TextTensor("test text")

    # Call backward with empty grad
    tensor.backward("")

    # Verify that text_grad remains empty
    assert tensor.text_grad == ""

    # Verify that no gradient was propagated to parents
    assert len(tensor.parents) == 0


def test_backward_with_parent_requires_grad_false(mock_agent):
    """Test backward pass when parent tensor has requires_grad=False."""
    # Create parent tensor with requires_grad=False
    parent_tensor = TextTensor("parent text", requires_grad=False)

    # Create child tensor with requires_grad=True and parent
    child_tensor = TextTensor("child text", parents=[parent_tensor], requires_grad=True)

    # Perform backward pass
    child_tensor.backward("some gradient")

    # Verify child tensor got the gradient
    assert child_tensor.text_grad == "some gradient"

    # Verify parent tensor did not get updated
    assert parent_tensor.text_grad == ""
    # Verify agent was not called for parent
    assert not mock_agent.run_sync.called


def test_backward_with_parent_requires_grad_true(mock_agent):
    """Test backward pass when parent tensor has requires_grad=True."""
    # Create parent tensor with requires_grad=True
    parent_tensor = TextTensor("parent text", requires_grad=True)

    # Create child tensor with requires_grad=True and parent
    child_tensor = TextTensor("child text", parents=[parent_tensor], requires_grad=True)

    # Mock the agent's response for gradient calculation
    mock_agent.run_sync.return_value.data = "parent gradient"

    # Perform backward pass
    child_tensor.backward("some gradient")

    # Verify child tensor got the gradient
    assert child_tensor.text_grad == "some gradient"

    # Verify parent tensor got updated with calculated gradient
    assert parent_tensor.text_grad == "parent gradient"

    # Verify agent was called with correct arguments
    mock_agent.run_sync.assert_called_once()
    call_args = mock_agent.run_sync.call_args[0][0]
    assert "parent text" in call_args
    assert "child text" in call_args
    assert "some gradient" in call_args


def test_calc_grad(mock_agent):
    """Test the calc_grad method."""
    # Create a tensor
    tensor = TextTensor("test text")

    # Mock the agent's response
    mock_agent.run_sync.return_value.data = "improved input"

    # Call calc_grad
    result = tensor.calc_grad("input text", "output text", "feedback")

    # Verify the result
    assert result == "improved input"

    # Verify agent was called with correct arguments
    mock_agent.run_sync.assert_called_once()
    call_args = mock_agent.run_sync.call_args[0][0]
    assert "input text" in call_args
    assert "output text" in call_args
    assert "feedback" in call_args
    assert "How should I improve the input" in call_args


def test_str():
    """Test the __str__ method."""
    # Create a tensor with some text
    tensor = TextTensor("test text")

    # Test string representation
    assert str(tensor) == "test text"

    # Test with different text
    tensor = TextTensor("another text")
    assert str(tensor) == "another text"
