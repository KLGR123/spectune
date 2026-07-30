"""Chinese prompt and phrasing assets for query construction.

Two rendering paths share this module. The deterministic path stitches an
information block together from the pools below, which is what a run produces
when LLM rewriting is skipped (``raw_query_ratio``) or no model is reachable.
The LLM path sends that same block to a chat model with
:data:`QUERY_REWRITE_SYSTEM_PROMPT`, whose whole job is to wrap it in the way a
hurried bench chemist would actually type it.

The prompts are deliberately strict about one thing: the model may only add a
short lead-in and closing ask. Spectra, formulas, SMILES, and reaction details
must survive character-for-character, since they are the supervision signal.
:func:`~spectune.llm.LlmClient` output is checked against that rule
by the caller and rejected back to the template text if the model drifts.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

JsonDict = dict[str, Any]

QUERY_REWRITE_SYSTEM_PROMPT = """你在为一个谱图解析应用构造训练数据。用户的最终诉求都是：根据给出的谱图（以及可能补充的其他线索）推断分子结构。

你的任务：把给定的信息块改写成一句真实用户会直接敲进对话框的提问。

硬性要求：
1. 信息块里的谱图数据、分子式、SMILES、化合物名称、反应物、条件、片段名称必须原样保留，数字、括号、逗号、大小写一个都不能改，也不要重新排版或补全。
2. 你只能在信息前后加口语化的引导语和诉求，前后缀加起来不超过 30 个字。
3. 语气像正在赶实验进度的研究生或研发工程师：直接、口语、偶尔省略主语。不要客套话、不要自我介绍、不要 emoji、不要 markdown 标题或列表、不要解释你做了什么。
4. 只输出最终的提问文本本身，不要加引号。
5. 用中文写引导语（信息块本身是英文或化学式的，保持原样）。

示例：
[信息块] 1H NMR (400 MHz, CDCl3): 7.35 (d, J = 8.1 Hz, 2H), 2.41 (s, 3H)
[输出] 帮我看下这个谱图是什么结构 1H NMR (400 MHz, CDCl3): 7.35 (d, J = 8.1 Hz, 2H), 2.41 (s, 3H)

[信息块] 13C NMR (101 MHz, DMSO): 168.2, 143.1, 128.9, 21.5
分子式：C8H8O2
[输出] 这个化合物的结构能推出来吗？13C NMR (101 MHz, DMSO): 168.2, 143.1, 128.9, 21.5，分子式是 C8H8O2

[信息块] 1H NMR (300 MHz, CDCl3): 8.02 (s, 1H), 3.88 (s, 3H)
可能含有三氟甲基
[输出] 解一下，1H NMR (300 MHz, CDCl3): 8.02 (s, 1H), 3.88 (s, 3H)，感觉里面有个三氟甲基"""

FOLLOWUP_REWRITE_SYSTEM_PROMPT = """你在为一个谱图解析应用构造多轮训练数据。用户上一轮已经贴了谱图并让模型推结构，现在要发第二条消息补充新线索。

你的任务：把给定的信息块改写成这条追加消息。

硬性要求：
1. 信息块里的分子式、SMILES、名称、反应物、条件、片段名称必须原样保留，一个字符都不能改。
2. 只能在前后加很短的口语化衔接，前后缀加起来不超过 25 个字，整条消息要明显比首轮短。
3. 口吻是“忘了说”、“补充一下”、“对了”这类补充信息的语气，可以带一点不确定（比如“大概这样”、“不太确定对不对”）。不要重复谱图、不要客套、不要 emoji、不要 markdown。
4. 只输出这条追加消息本身，不要加引号。

示例：
[信息块] 分子式：C9H11NO4S
[输出] 补充下，分子式是 C9H11NO4S，再帮我看看

[信息块] 可能是 CNS(=O)(=O)c1cccc(C(=O)OC)c1
[输出] 我怀疑是 CNS(=O)(=O)c1cccc(C(=O)OC)c1，不太确定"""

# Deterministic phrasing pools

FIRST_TURN_LEADS = (
    "解析下列分子结构：",
    "帮我看下这个谱图对应什么结构：",
    "这个谱图能推出结构吗？",
    "这组数据是什么化合物：",
    "麻烦解一下这个谱：",
    "下面是实验测的谱图，帮我推一下结构：",
    "这个化合物的结构是什么？谱图数据如下：",
)

FIRST_TURN_TAILS = (
    "",
    "",
    "给出可能的 SMILES。",
    "麻烦给出可能的结构。",
    "最好能说下判断依据。",
    "谢谢。",
)

FOLLOWUP_LEADS = (
    "补充一下，",
    "忘了说，",
    "对了，",
    "另外，",
    "补个信息，",
)

FOLLOWUP_TAILS = (
    "",
    "",
    "再帮我看看。",
    "这样能确定吗？",
    "再确认下。",
)

UNCERTAIN_PREFIXES = (
    "怀疑是",
    "应该是",
    "可能是",
    "初步猜是",
    "不确定，可能是",
)

UNCERTAIN_SUFFIXES = (
    "，不太确定",
    "，目前是这样",
    "，基本上是这样",
    "，帮我核实下",
    "",
)


def _choice(rng: random.Random, options: Sequence[str]) -> str:
    return rng.choice(list(options))


def render_formula_block(formula: str) -> str:
    return f"分子式：{formula}"


def render_structure_block(value: str, kind: str, rng: random.Random) -> str:
    """Phrase a structure guess with the hedged tone real users apply to it."""
    label = {"smiles": "SMILES", "name_zh": "中文名", "name_en": "英文名"}.get(kind, "结构")
    prefix = _choice(rng, UNCERTAIN_PREFIXES)
    suffix = _choice(rng, UNCERTAIN_SUFFIXES)
    if kind == "smiles":
        return f"{prefix} {value}{suffix}"
    return f"{prefix}{value}（{label}）{suffix}"


def render_reaction_block(items: Sequence[JsonDict], rng: random.Random, *, uncertain: bool) -> str:
    """Describe how the compound was (or may have been) made."""
    lines: list[str] = []
    for item in items:
        reactants = "、".join(item.get("reactants") or [])
        conditions = item.get("conditions") or ""
        name = item.get("name") or ""
        parts = [f"原料：{reactants}"] if reactants else []
        if conditions:
            parts.append(f"条件：{conditions}")
        if name:
            parts.append(f"反应类型：{name}")
        if parts:
            lines.append("，".join(parts))
    if not lines:
        return ""
    head = "如下是化合物的大致合成路线" if uncertain else "化合物合成信息如下"
    body = "；".join(lines)
    tail = _choice(rng, UNCERTAIN_SUFFIXES) if uncertain else ""
    return f"{head}：{body}{tail}"


def render_polymer_block(classes: Sequence[JsonDict]) -> str:
    if not classes:
        return ""
    described = "、".join(f"{item['zh']}（{item['polymerization_zh']}）" for item in classes[:2])
    return f"这批样品是做高分子的，可能涉及{described}"


def render_fragment_block(fragments: Sequence[JsonDict], rng: random.Random, *, uncertain: bool) -> str:
    if not fragments:
        return ""
    described = "、".join(str(item.get("zh") or item.get("en") or "") for item in fragments if item)
    if not described:
        return ""
    if uncertain:
        return f"{_choice(rng, ('结构里应该有', '含有', '大概率带'))}{described}{_choice(rng, UNCERTAIN_SUFFIXES)}"
    return f"已知结构中含有{described}"


def render_first_turn(spectrum_text: str, information: str, rng: random.Random) -> str:
    lead = _choice(rng, FIRST_TURN_LEADS)
    tail = _choice(rng, FIRST_TURN_TAILS)
    body = spectrum_text if not information else f"{spectrum_text}\n{information}"
    return f"{lead}{body}\n{tail}".strip()


def render_followup(information: str, rng: random.Random) -> str:
    lead = _choice(rng, FOLLOWUP_LEADS)
    # An information block that already hedges ("...，帮我核实下") needs no
    # closing ask; stacking both reads like two people talking.
    hedged = information.endswith(tuple(suffix for suffix in UNCERTAIN_SUFFIXES if suffix))
    tail = "" if hedged else _choice(rng, FOLLOWUP_TAILS)
    return f"{lead}{information}{('。' if not information.endswith(('。', '？', '，')) else '')}{tail}".strip()


__all__ = [
    "FOLLOWUP_REWRITE_SYSTEM_PROMPT",
    "QUERY_REWRITE_SYSTEM_PROMPT",
    "render_first_turn",
    "render_followup",
    "render_formula_block",
    "render_fragment_block",
    "render_polymer_block",
    "render_reaction_block",
    "render_structure_block",
]
