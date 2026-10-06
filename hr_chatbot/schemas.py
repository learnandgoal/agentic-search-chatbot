"""Tool argument schemas (what the model sees) and the structured final answer schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SUBMIT_NAME = "submit_answer"


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=300, description="Keywords or a natural-language description of what to find.")
    top_k: int = Field(default=5, ge=1, description="How many hits to return (max 10; larger values are capped).")
    exclude_ids: list[int] = Field(
        default_factory=list,
        description="chunk_ids to leave out, e.g. ones you already looked at. Previously seen chunks are always excluded.",
    )


class OpenArgs(BaseModel):
    document_id: str = Field(description="A document_id returned by search.")


class NavigateArgs(BaseModel):
    chunk_id: int = Field(description="The chunk you are currently at (from search, navigate, read or grep).")
    direction: Literal["next", "previous", "next_section", "previous_section"] = "next"
    steps: int = Field(default=1, ge=1, description="For next/previous: how many chunks to move (max 3).")


class ReadArgs(BaseModel):
    chunk_id: int | None = Field(default=None, description="A chunk_id returned by search/grep/navigate/open (never a guess or 0). Reads this chunk plus `before`/`after` neighbours. Use this OR document_id+page_or_section, not both.")
    document_id: str | None = Field(default=None, description="Use with page_or_section to read a whole page or section.")
    page_or_section: str | None = Field(default=None, description="Exact label as shown by open/search, e.g. 'Page 2 - Experience'.")
    before: int = Field(default=0, ge=0, description="Neighbouring chunks before chunk_id (max 5).")
    after: int = Field(default=0, ge=0, description="Neighbouring chunks after chunk_id (max 5).")
    max_chars: int = Field(default=3000, ge=200, description="Character budget for the returned text (max 6000).")

    @model_validator(mode="after")
    def _one_mode(self) -> "ReadArgs":
        if self.chunk_id is None and not (self.document_id and self.page_or_section):
            raise ValueError("Provide chunk_id, or document_id together with page_or_section.")
        return self


class GrepArgs(BaseModel):
    document_id: str = Field(description="The single document to search inside.")
    pattern: str = Field(min_length=1, description="Exact word or phrase (case-insensitive literal, not a regex), max 100 chars.")
    whole_word: bool = Field(default=False, description="Only match the pattern as a whole word.")
    max_results: int = Field(default=10, ge=1, description="Maximum matches to return (max 20).")
    context_chars: int = Field(default=80, ge=0, description="Characters of context around each match (max 200).")


class EvidenceRef(BaseModel):
    chunk_id: int = Field(description="chunk_id of a chunk whose text you received from read, grep or navigate (not search or open).")
    quote: str = Field(
        max_length=400,
        description="Short passage copied verbatim from the text that read, grep or navigate returned for that chunk.",
    )


class SubmitAnswer(BaseModel):
    intent: Literal["qa", "unsupported"] = Field(
        description="'qa' for questions about this candidate, this job, or their fit. 'unsupported' for anything else."
    )
    response: str = Field(
        description="The answer for the recruiter, in markdown. For the initial briefing use three headings: "
        "Candidate summary, Job summary, Fit analysis."
    )
    evidence: list[EvidenceRef] = Field(
        default_factory=list, description="Citations backing the factual claims in the response. Empty if intent is 'unsupported'."
    )
