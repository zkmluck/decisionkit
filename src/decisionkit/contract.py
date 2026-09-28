"""The typed decision contract: a state plus typed questions in, distributions out.

Nothing here generates text. A question is one of three primitives, and every
primitive returns the whole candidate distribution rather than only the winner.
Transport ids stay out of the model input, so they can never leak into a score.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

API_VERSION = 'decisionkit.v1'
PRIMITIVES = ('boolean', 'choice', 'score')
BOOLEAN_KEYS = ('true', 'false')
MAX_REQUESTS = 32
MAX_QUESTIONS = 128
MAX_PATHS = 1024


class ContractError(ValueError):
    """A request that does not satisfy the contract."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Question(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    type: str = 'choice'
    question: str = Field(min_length=1, max_length=4000)
    criteria: dict[str, str] | None = None
    options: dict[str, str] | list[str] | None = None
    levels: list[str] | None = None


class Request(StrictModel):
    id: str | None = Field(default=None, max_length=200)
    state: Any = None
    questions: list[Question] = Field(min_length=1, max_length=MAX_QUESTIONS)


class PreparedQuestion(StrictModel):
    id: str
    type: str
    text: str
    keys: list[str]
    candidates: list[str]


class PreparedRequest(StrictModel):
    id: str
    state: str
    questions: list[PreparedQuestion]


class PreparedBatch(StrictModel):
    requests: list[PreparedRequest]
    min_probability: float
    min_margin: float


def semantic_text(value: Any, name: str) -> str:
    """Text stays text; a structured state becomes canonical JSON."""
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(',', ':'), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ContractError(f'{name} must contain valid JSON') from exc
    raise ContractError(f'{name} must be nonempty text, an object or an array')


def _candidates(question: Question) -> tuple[list[str], list[str]]:
    if question.type not in PRIMITIVES:
        raise ContractError('type must be boolean, choice or score')
    if question.type == 'boolean':
        criteria = question.criteria or {}
        if not isinstance(criteria, dict):
            raise ContractError('criteria must be an object')
        return list(BOOLEAN_KEYS), [semantic_text(criteria[key], f'{key} criterion')
                                    if key in criteria else key.upper() for key in BOOLEAN_KEYS]
    if question.type == 'choice':
        options = question.options
        if isinstance(options, dict):
            keys = list(options)
            texts = [semantic_text(value, 'option') for value in options.values()]
        elif isinstance(options, list):
            keys = [str(index) for index in range(len(options))]
            texts = [semantic_text(value, 'option') for value in options]
        else:
            raise ContractError('choice.options must be an object or an array')
        if not 2 <= len(keys) <= 255:
            raise ContractError('choice requires 2..255 candidates')
        if any(not isinstance(key, str) or not key for key in keys):
            raise ContractError('option ids must be nonempty strings')
        return keys, texts
    levels = question.levels
    if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
        raise ContractError('score.levels must be a list of 2..10 ordered descriptions')
    return [str(index) for index in range(len(levels))], [semantic_text(value, 'level') for value in levels]


def prepare(payload: Any) -> PreparedBatch:
    """Validate a payload and freeze it into the shape the model consumes."""
    if not isinstance(payload, dict):
        raise ContractError('request must be an object')
    envelope = {key: value for key, value in payload.items()
                if key not in ('requests', 'min_probability', 'min_margin')}
    raw = payload.get('requests', [envelope])
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_REQUESTS:
        raise ContractError(f'requests must contain 1..{MAX_REQUESTS} states')
    requests = []
    total_questions = total_paths = 0
    for index, item in enumerate(raw):
        try:
            request = Request.model_validate(item)
        except ValidationError as exc:
            raise ContractError(f'request {index}: {exc.errors()[0]["msg"]}') from exc
        if request.state is None:
            raise ContractError(f'request {index}: state is required')
        state = semantic_text(request.state, 'state')
        rows, seen = [], set()
        for question in request.questions:
            if question.id in seen:
                raise ContractError(f'question ids must be unique within a state: {question.id!r}')
            seen.add(question.id)
            keys, texts = _candidates(question)
            if len(set(texts)) != len(texts):
                raise ContractError(f'question {question.id!r}: candidate descriptions must be distinct')
            rows.append(PreparedQuestion(id=question.id, type=question.type,
                                         text=semantic_text(question.question, 'question'),
                                         keys=keys, candidates=texts))
            total_questions += 1
            total_paths += len(keys)
        requests.append(PreparedRequest(id=request.id or str(index), state=state, questions=rows))
    if total_questions > MAX_QUESTIONS:
        raise ContractError(f'batch exceeds {MAX_QUESTIONS} questions')
    if total_paths > MAX_PATHS:
        raise ContractError(f'batch exceeds {MAX_PATHS} candidate paths')
    min_probability = payload.get('min_probability', 0.6)
    min_margin = payload.get('min_margin', 0.05)
    for name, value in (('min_probability', min_probability), ('min_margin', min_margin)):
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise ContractError(f'{name} must be a number between 0 and 1')
    return PreparedBatch(requests=requests, min_probability=float(min_probability),
                         min_margin=float(min_margin))
