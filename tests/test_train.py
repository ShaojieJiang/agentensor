"""Test module for the Trainer class."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic_evals import Dataset
from pydantic_graph import Graph
from agentensor.module import AgentModule, ModuleState
from agentensor.optim import Optimizer
from agentensor.tensor import TextTensor
from agentensor.train import Trainer


@pytest.fixture
def mock_graph():
    """Create a mock graph for testing."""
    mock_graph = MagicMock(spec=Graph)
    return mock_graph


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    mock_dataset = MagicMock(spec=Dataset)
    return mock_dataset


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    mock_optimizer = MagicMock(spec=Optimizer)
    mock_optimizer.params = []  # Add params attribute
    return mock_optimizer


@pytest.fixture
def mock_module_class():
    """Create a mock module class for testing."""

    class MockModule(AgentModule):
        def __init__(self):
            super().__init__()

        async def run(self, state: ModuleState) -> ModuleState:
            return state

    return MockModule


@pytest.mark.asyncio
async def test_trainer_initialization(
    mock_graph, mock_dataset, mock_optimizer, mock_module_class
):
    """Test Trainer initialization."""
    trainer = Trainer(
        graph=mock_graph,
        start_node=mock_module_class,
        dataset=mock_dataset,
        optimizer=mock_optimizer,
        epochs=10,
        stop_threshold=0.95,
    )

    assert trainer.graph == mock_graph
    assert trainer.start_node == mock_module_class
    assert trainer.dataset == mock_dataset
    assert trainer.optimizer == mock_optimizer
    assert trainer.epochs == 10
    assert trainer.stop_threshold == 0.95


@pytest.mark.asyncio
async def test_trainer_step(
    mock_graph, mock_dataset, mock_optimizer, mock_module_class
):
    """Test the step method of Trainer."""
    # Setup
    trainer = Trainer(
        graph=mock_graph,
        start_node=mock_module_class,
        dataset=mock_dataset,
        optimizer=mock_optimizer,
        epochs=10,
    )

    # Mock the graph's run method
    mock_graph.run = AsyncMock()
    state = ModuleState(input=TextTensor("test input"))
    state.output = TextTensor("test output")
    mock_graph.run.return_value = state

    # Test step
    input_tensor = TextTensor("test input")
    result = await trainer.step(input_tensor)

    # Verify
    assert isinstance(result, TextTensor)
    assert result.text == "test output"
    mock_graph.run.assert_called_once()


def test_trainer_train(mock_graph, mock_dataset, mock_optimizer, mock_module_class):
    """Test the train method of Trainer."""
    # Setup
    trainer = Trainer(
        graph=mock_graph,
        start_node=mock_module_class,
        dataset=mock_dataset,
        optimizer=mock_optimizer,
        epochs=2,
    )

    # Mock dataset evaluation
    mock_report = MagicMock()
    mock_report.cases = []
    mock_report.averages.return_value.assertions = 0.96  # Above stop threshold
    mock_dataset.evaluate_sync.return_value = mock_report

    # Run training
    trainer.train()

    # Verify
    assert mock_dataset.evaluate_sync.call_count == 1
    assert mock_optimizer.step.call_count == 1
    assert mock_optimizer.zero_grad.call_count == 1


def test_trainer_train_with_failed_cases(
    mock_graph, mock_dataset, mock_optimizer, mock_module_class
):
    """Test the train method with failed cases that need backward pass."""
    # Setup
    trainer = Trainer(
        graph=mock_graph,
        start_node=mock_module_class,
        dataset=mock_dataset,
        optimizer=mock_optimizer,
        epochs=2,
    )

    # Create a mock case with failed assertions
    mock_case = MagicMock()
    mock_case.output = TextTensor("test output", requires_grad=True)
    mock_case.assertions = {
        "test1": MagicMock(value=False, reason="error1"),
        "test2": MagicMock(value=True, reason=None),
    }

    # Mock dataset evaluation
    mock_report = MagicMock()
    mock_report.cases = [mock_case]
    mock_report.averages.return_value.assertions = 0.5  # Below stop threshold
    mock_dataset.evaluate_sync.return_value = mock_report

    # Run training
    trainer.train()

    # Verify
    assert mock_dataset.evaluate_sync.call_count == 2  # Called for each epoch
    assert mock_optimizer.step.call_count == 2
    assert mock_optimizer.zero_grad.call_count == 2
    assert mock_case.output.text_grad == "error1"  # Verify backward pass was called


def test_trainer_early_stopping(
    mock_graph, mock_dataset, mock_optimizer, mock_module_class
):
    """Test early stopping when performance threshold is reached."""
    # Setup
    trainer = Trainer(
        graph=mock_graph,
        start_node=mock_module_class,
        dataset=mock_dataset,
        optimizer=mock_optimizer,
        epochs=10,
        stop_threshold=0.95,
    )

    # Mock dataset evaluation with high performance
    mock_report = MagicMock()
    mock_report.cases = []
    mock_report.averages.return_value.assertions = 0.96  # Above stop threshold
    mock_dataset.evaluate_sync.return_value = mock_report

    # Run training
    trainer.train()

    # Verify early stopping
    assert mock_dataset.evaluate_sync.call_count == 1  # Only one epoch before stopping
    assert mock_optimizer.step.call_count == 1
    assert mock_optimizer.zero_grad.call_count == 1


def test_trainer_train_with_no_losses(
    mock_graph, mock_dataset, mock_optimizer, mock_module_class
):
    """Test the train method when all assertions pass and there are no losses."""
    # Setup
    trainer = Trainer(
        graph=mock_graph,
        start_node=mock_module_class,
        dataset=mock_dataset,
        optimizer=mock_optimizer,
        epochs=2,
    )

    # Create a mock case with all passing assertions
    mock_case = MagicMock()
    mock_case.output = TextTensor("test output", requires_grad=True)
    mock_case.assertions = {
        "test1": MagicMock(value=True, reason=None),
        "test2": MagicMock(value=True, reason=None),
    }

    # Mock dataset evaluation
    mock_report = MagicMock()
    mock_report.cases = [mock_case]
    mock_report.averages.return_value.assertions = 0.5  # Below stop threshold
    mock_dataset.evaluate_sync.return_value = mock_report

    # Run training
    trainer.train()

    # Verify
    assert mock_dataset.evaluate_sync.call_count == 2  # Called for each epoch
    assert mock_optimizer.step.call_count == 2
    assert mock_optimizer.zero_grad.call_count == 2
    assert (
        mock_case.output.text_grad == ""
    )  # No backward pass since all assertions passed
