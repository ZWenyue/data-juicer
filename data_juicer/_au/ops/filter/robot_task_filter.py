# -*- coding: utf-8 -*-
"""Filter LeRobot tasks by task-level text metadata."""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Sequence

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields

OP_NAME = "robot_task_filter"

_SKILL_KEYWORDS = {
    "open_door": (
        "开门",
        "打开门",
        "open door",
        "open the door",
    ),
}

_SKILL_PATTERNS = {
    "open_door": (
        re.compile(r"\b(?:open|opening|push|pull)\b.{0,40}\bdoor\b", re.IGNORECASE),
        re.compile(r"(?:打开|推开|拉开|开).{0,12}门"),
    ),
}


def resolve_skill_keywords(
    skill: str = "",
    include_keywords: Sequence[str] | None = None,
) -> List[str]:
    """Combine a built-in skill vocabulary with user-provided keywords."""
    keywords: List[str] = []
    if skill:
        if skill not in _SKILL_KEYWORDS:
            supported = ", ".join(sorted(_SKILL_KEYWORDS))
            raise ValueError(f"Unknown robot task skill {skill!r}; supported skills: {supported}")
        keywords.extend(_SKILL_KEYWORDS[skill])
    keywords.extend(str(word) for word in (include_keywords or ()))

    result: List[str] = []
    seen = set()
    for word in keywords:
        word = word.strip()
        if word and word not in seen:
            seen.add(word)
            result.append(word)
    if not result:
        raise ValueError("skill or include_keywords must provide at least one keyword")
    return result


def _flatten_text(values: object) -> Iterable[str]:
    if values is None:
        return
    if isinstance(values, str):
        yield values
    elif isinstance(values, dict):
        for value in values.values():
            yield from _flatten_text(value)
    elif isinstance(values, (list, tuple, set)):
        for value in values:
            yield from _flatten_text(value)
    else:
        yield str(values)


def _normalize(text: str, case_sensitive: bool) -> str:
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if case_sensitive else text.casefold()


@OPERATORS.register_module(OP_NAME)
class RobotTaskFilter(Filter):
    """Keep a complete LeRobot task when its task-level text matches keywords.

    The input sample represents one task directory and normally contains
    ``task_name`` and may contain ``task_labels`` loaded from
    ``meta/tasks.jsonl``. A match in either source keeps the complete task
    directory; no episode or frame data is inspected or split.
    """

    def __init__(
        self,
        skill: str = "",
        include_keywords: Sequence[str] | None = None,
        exclude_keywords: Sequence[str] | None = None,
        text_fields: Sequence[str] = ("task_name", "task_labels"),
        case_sensitive: bool = False,
        report_field: str = "robot_task_filter_report",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.skill = skill
        self.include_keywords = resolve_skill_keywords(skill, include_keywords)
        self.exclude_keywords = [str(word).strip() for word in (exclude_keywords or ()) if str(word).strip()]
        self.text_fields = tuple(text_fields)
        self.case_sensitive = bool(case_sensitive)
        self.report_field = report_field

    def _matched_keywords(self, texts: Sequence[str], keywords: Sequence[str]) -> List[str]:
        normalized_texts = [_normalize(text, self.case_sensitive) for text in texts if text.strip()]
        matched = []
        for keyword in keywords:
            normalized_keyword = _normalize(keyword, self.case_sensitive)
            if any(normalized_keyword in text for text in normalized_texts):
                matched.append(keyword)
        return matched

    def _skill_matches(self, texts: Sequence[str]) -> List[str]:
        if not self.skill:
            return []
        for pattern in _SKILL_PATTERNS.get(self.skill, ()):
            if any(pattern.search(_normalize(text, True)) for text in texts if text.strip()):
                return [f"skill:{self.skill}"]
        return []

    def compute_stats_single(self, sample, context=False):
        stats = sample.setdefault(Fields.stats, {})
        meta = sample.setdefault(Fields.meta, {})

        texts: List[str] = []
        for field in self.text_fields:
            texts.extend(_flatten_text(sample.get(field)))

        include_matches = self._matched_keywords(texts, self.include_keywords)
        include_matches.extend(self._skill_matches(texts))
        include_matches = list(dict.fromkeys(include_matches))
        exclude_matches = self._matched_keywords(texts, self.exclude_keywords)
        keep = bool(include_matches) and not exclude_matches

        stats["robot_task_filter_keep"] = bool(keep)
        stats["robot_task_filter_match_count"] = int(len(include_matches))
        meta[self.report_field] = json.dumps(
            {
                "keep": bool(keep),
                "skill": self.skill,
                "include_matches": include_matches,
                "exclude_matches": exclude_matches,
                "text_fields": list(self.text_fields),
            },
            ensure_ascii=False,
        )
        return sample

    def process_single(self, sample):
        return sample.get(Fields.stats, {}).get("robot_task_filter_keep", False)
