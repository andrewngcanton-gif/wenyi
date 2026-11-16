"""Deterministic replacement for the external LLM boundary.

This module is loaded inside a child process after the selected code source has
been placed first on ``sys.path``. It replaces only ``build_client`` in
``wenyi_core.llm.factory``; orchestration, CLI, receipts and storage remain the
real product code. Replies are synthetic and must never be presented as model
quality evidence.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

LANGUAGE = "en"
TARGET_LANGUAGE = "zh"

_TRANSLATIONS = {
    "Mira carried the old map to the pier.": "米拉带着旧地图走向码头。",
    "The lantern glowed over the quiet harbor.": "灯笼映照着寂静的港湾。",
    "Every evening the keeper lit the lamp.": "每晚守灯人都会点起灯。",
    "A gull circled twice, then vanished.": "一只海鸥盘旋两圈后消失了。",
    '"The tide is turning," Mira said quietly.': "“潮水正在转向。”米拉轻声说。",
    '"We should leave before the fog arrives," said the keeper.': "“我们该在雾来之前离开。”守灯人说。",
    "Mira wondered whether the old path still led to the spring.": "米拉想知道那条旧路是否仍通向泉水。",
    "She remembered the winter when the harbor froze.": "她记得港湾结冰的那个冬天。",
    "Red ink marked the hidden spring on the map.": "红墨水在地图上标出了隐藏的泉水。",
    "The eastern gate opened at dawn.": "东门在黎明时打开。",
    "Bells rang across the water.": "钟声越过水面传来。",
    "The keeper counted the boats and wrote each name in his book.": "守灯人数着船只，把每个名字写进本子。",
    "Lantern light on water; a map folded in her hand; the keeper waits for morning.": (
        "水面灯影；地图折在她手中；守灯人等候清晨。"
    ),
}

_NUMBERED = re.compile(r"^\[(\d+)\]\s?(.*)$", re.MULTILINE)

_CHAPTERS = {
    "Chapter 1: The Harbor": "第一章：港湾",
    "Chapter 2: The Map": "第二章：地图",
    "Chapter 3: The Keeper": "第三章：守灯人",
    "Chapter 4: The Gull": "第四章：海鸥",
    "Chapter 5: The Fog": "第五章：雾",
    "Chapter 6: The Spring": "第六章：泉水",
    "Chapter 7: The Gate": "第七章：门",
    "Chapter 8: The Bells": "第八章：钟声",
    "Chapter 9: The Tide": "第九章：潮水",
    "Chapter 10: Morning": "第十章：清晨",
}


def translate_text(text: str) -> str:
    """Translate known fixture sentences and mark unknown text deterministically."""
    stripped = text.strip()
    if stripped in _TRANSLATIONS:
        return _TRANSLATIONS[stripped]
    if stripped in _CHAPTERS:
        return _CHAPTERS[stripped]
    if stripped.startswith("#"):
        return "第" + stripped.lstrip("#").strip() + "章"
    if stripped.startswith("Chapter "):
        return "第" + stripped.removeprefix("Chapter ") + "章"
    return "译文：" + stripped


def polish_text(text: str) -> str:
    stripped = text.strip()
    if stripped in _TRANSLATIONS:
        return _TRANSLATIONS[stripped]
    return "润色：" + stripped


def _numbered(text: str) -> list[str]:
    return [item.strip() for _, item in _NUMBERED.findall(text)]


def _count(text: str) -> int:
    return len(_NUMBERED.findall(text))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _last_assistant_count(messages) -> int | None:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except json.JSONDecodeError:
            return None
        items = payload.get("translations") if isinstance(payload, dict) else None
        return len(items) if isinstance(items, list) else None
    return None


def handler(messages, tier: str, json_mode: bool) -> str:
    """Return a schema-compatible, deterministic reply for each agent task."""
    system = str(messages[0].get("content", "")) if messages else ""
    user = str(messages[-1].get("content", "")) if messages else ""
    lowered = system.lower()

    if "language detector" in lowered:
        return _json({"language": LANGUAGE})
    if "pre-translation analyst" in lowered:
        return _json(
            {
                "genre": "literary",
                "tone": "quiet",
                "style_guide": "Use short, concrete sentences and preserve dialogue.",
                "narration": "close third person",
                "pacing": "measured",
                "dialogue_style": "restrained",
                "characters": [
                    {"source": "Mira", "gender": "female"},
                    {"source": "the keeper", "gender": "unknown"},
                ],
            }
        )
    if "chapter title translator" in lowered:
        titles = _numbered(user)
        return _json({"titles": [translate_text(title) for title in titles]})
    if "literary translator" in lowered or "academic translator" in lowered:
        if "Polish the translations from your previous JSON response" in user:
            count = _last_assistant_count(messages[:-1])
            if count is None:
                count = _count(user)
            return _json({"polished": [f"润色译文{i}" for i in range(count)]})
        sources = _numbered(user)
        return _json({"translations": [translate_text(source) for source in sources]})
    if "prose editor" in lowered or "polisher" in lowered:
        targets = _numbered(user)
        if not targets:
            count = _last_assistant_count(messages)
            targets = [f"译文{i}" for i in range(count or 0)]
        return _json({"polished": [polish_text(target) for target in targets]})
    if "translation reviewer" in lowered or "reviewer" in lowered:
        return _json(
            {
                "issues": [],
                "reviewed_segments": _count(user),
                "complete": True,
            }
        )
    if "terminology" in lowered and "extractor" in lowered:
        terms = []
        if "Mira" in user:
            terms.append({"source": "Mira", "target": "米拉", "type": "person", "gender": "female"})
        if "lantern" in user.lower():
            terms.append({"source": "lantern", "target": "灯笼", "type": "object"})
        return _json({"terms": terms})
    if "chapter digest writer" in lowered:
        return "本章梗概：米拉与守灯人在港湾寻找通往泉水的旧路。"
    if "whole-book synopsis writer" in lowered:
        return "全书概览：一张旧地图引导米拉在港湾寻找隐藏的泉水。"
    if "terminology" in lowered:
        return _json({"terms": []})
    return "{}" if json_mode else ""


def _synthetic_usage(messages, reply: str) -> tuple[int, int]:
    prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
    return max(1, math.ceil(prompt_chars / 4)), max(1, math.ceil(len(reply) / 4))


def build_client(config, *, credentials=None):
    """Build the offline client with the same public interface as the factory."""
    from wenyi_core.llm.providers.fake import FakeClient
    from wenyi_core.llm.usage import UsageSample

    class OfflineClient(FakeClient):
        def complete(self, messages, *, operation, json_mode=False, max_tokens=None):
            reply = super().complete(
                messages,
                operation=operation,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )
            prompt_tokens, completion_tokens = _synthetic_usage(messages, reply)
            route = self.routes[operation]
            self.usage.record(
                route.tier or "direct",
                UsageSample(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    cache_miss_tokens=prompt_tokens,
                ),
                operation,
            )
            return reply

    return OfflineClient(handler, config=config.llm)
