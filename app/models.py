from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field

class SchemaInfo(BaseModel):
    columns: list[str] = Field(description = "Column names in courrent order")
    dtypes: dict[str, str] = Field(description="Column name → dtype, e.g. 'object', 'int64'")
    shape: tuple[int, int] = Field(description="(num_rows, num_cols)")
    
class Observation(BaseModel):
    task_id: str = Field(description="Which task is running: 'easy' | 'medium' | 'hard'")
    step: int = Field(description="Current step number (0-indexed)")
    max_steps: int = Field(description="Maximum allowed steps for this task")
    df_preview: list[dict[str, Any]] = Field(description="First 5 rows of the DataFrame as records")
    schema_info: SchemaInfo = Field(description="Column names, dtypes, and shape")
    null_count: dict[str, int] = Field(description="Number of nulls per column")
    duplicate_count: int = Field(description="Number of fully duplicate rows")
    schema_errors: list[str] = Field(default_factory=list, description="Any structural issues detected")
    is_terminal: bool = Field(default=False, description="True when episode has ended")

class Action(BaseModel):
    operation: str = Field(description="One of: 'drop_nulls', 'normalize_dates', 'drop_duplicates', 'rename_column', 'validate_emails', 'lowercase_column', 'no_op'")
    column: Optional[str] = Field(default=None, description="Target column name. Some operations ignore this.")
    params: dict[str, Any] = Field(default_factory=dict, description="Extra parameters, e.g. {'new_name': 'user_id'}")

class RewardInfo(BaseModel):
    reward: float = Field(description="Dense reward for this step: score_delta - penalties")
    score: float = Field(description="Absolute grader score after this action. Range [0.0, 1.0]")
    done: bool = Field(description="True if the episode has ended")
    penalty: float = Field(default=0.0, description="Any penalty applied this step")
    info: dict[str, Any] = Field(default_factory=dict, description="Breakdown for debugging")


class StepResponse(BaseModel):
    observation: Observation
    reward_info: RewardInfo


class ResetResponse(BaseModel):
    observation: Observation
    message: str = "Episode reset successfully"