"""Deterministic hashed features for (state, question, candidate) triples.

A candidate is scored from three channels plus the interactions that make the
score depend on the state at all:

* ``s`` state tokens, ``q`` question tokens, ``c`` candidate tokens;
* ``i`` pairs of a candidate token with a state keyword, which is what lets the
  same candidate text score differently under different states.

Everything is a hash of a token, so the feature count is fixed, the order of the
candidates cannot matter, and no vocabulary file has to be shipped.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

DEFAULT_DIM = 1 << 15
MAX_STATE_KEYWORDS = 48
MAX_CANDIDATE_TOKENS = 12
STOPWORDS = frozenset(
    'a an and are as at be by for from has have if in is it its of on or that the '
    'this to was were will with not no do does'.split())
TOKEN = re.compile(r'[a-z0-9_]+')


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def state_keywords(state: str) -> list[str]:
    """Salient state tokens in first-appearance order, capped for bounded work."""
    seen, keywords = set(), []
    for token in tokenize(state):
        if len(token) < 3 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) == MAX_STATE_KEYWORDS:
            break
    return keywords


@lru_cache(maxsize=1 << 20)
def feature_index(name: str, dim: int) -> int:
    digest = hashlib.blake2b(name.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % dim


def candidate_features(state: str, question: str, candidate: str, dim: int = DEFAULT_DIM):
    """Sparse (indices, values) for one candidate; values are counts."""
    counts: dict[int, float] = {}
    for channel, text in (('s', state), ('q', question), ('c', candidate)):
        for token in tokenize(text):
            index = feature_index(f'{channel}:{token}', dim)
            counts[index] = counts.get(index, 0.0) + 1.0
    candidate_tokens = []
    for token in tokenize(candidate):
        if token not in candidate_tokens:
            candidate_tokens.append(token)
        if len(candidate_tokens) == MAX_CANDIDATE_TOKENS:
            break
    for token in candidate_tokens:
        for keyword in state_keywords(state):
            index = feature_index(f'i:{token}|{keyword}', dim)
            counts[index] = counts.get(index, 0.0) + 1.0
    indices = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    values = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    # Unit-norm rows keep long states from dominating short candidates.
    norm = float(np.sqrt((values ** 2).sum()))
    return indices, values / norm if norm else values


def question_features(state: str, question: str, candidates: list[str], dim: int = DEFAULT_DIM):
    """One sparse row per candidate, in the order the candidates were given."""
    return [candidate_features(state, question, candidate, dim) for candidate in candidates]


def feature_tokens(state: str, question: str, candidate: str) -> int:
    """How many hashed tokens a candidate contributes; reported as usage."""
    return (len(tokenize(state)) + len(tokenize(question)) + len(tokenize(candidate))
            + len(tokenize(candidate)) * len(state_keywords(state)))
