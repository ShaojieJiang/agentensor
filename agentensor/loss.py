"""Loss functions."""

from typing import Any
from pydantic import BaseModel, ConfigDict
from pydantic_ai import models
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output, judge_output
from agentensor.tensor import TextTensor


class LLMTensorJudgeConfig(BaseModel):
    """Configuration for LLMTensorJudge."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rubric: str
    model: models.Model | models.KnownModelName | None = None
    include_input: bool = True


class LLMTensorJudge(Evaluator[TextTensor, TextTensor, Any]):
    """LLM judge for text tensors.

    Adapted from pydantic_evals.evaluators.common.LLMJudge.
    """

    def __init__(
        self,
        rubric: str,
        model: models.Model | models.KnownModelName | None = None,
        include_input: bool = True,
    ):
        """Initialize the LLM tensor judge."""
        self.config = LLMTensorJudgeConfig(
            rubric=rubric,
            model=model,
            include_input=include_input,
        )

    @property
    def rubric(self) -> str:
        """Get the rubric."""
        return self.config.rubric

    @property
    def model(self) -> models.Model | models.KnownModelName | None:
        """Get the model."""
        return self.config.model

    @property
    def include_input(self) -> bool:
        """Get the include_input flag."""
        return self.config.include_input

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

    def build_serialization_arguments(self) -> dict[str, Any]:
        """Build serialization arguments."""
        result = {
            "rubric": self.rubric,
            "model": self.model,
            "include_input": self.include_input,
        }
        if (model := result.get("model")) and isinstance(model, models.Model):
            result["model"] = f"{model.system}:{model.model_name}"
        return result
