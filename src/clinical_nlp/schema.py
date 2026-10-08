"""Response schema for the extraction API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)


class Condition(BaseModel):
    condition: str = Field(min_length=1)
    span: tuple[int, int]
    negated: bool
    uncertain: bool

    @model_validator(mode="after")
    def span_is_ordered(self) -> "Condition":
        start, end = self.span
        if start < 0 or end <= start:
            raise ValueError("span must satisfy 0 <= start < end")
        return self


class ExtractResponse(BaseModel):
    conditions: list[Condition]
