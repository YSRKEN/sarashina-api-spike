"""Chat 以外の 4 モデル（guard・embedding・rerank・translate）の入力の組を読み、送る要求を並べる。

入力はモデルごとの JSONL（``<置き場>/<種類>.jsonl``、種類は ``guard``・``embedding``・
``rerank``・``translate``）。公開する ``inputs/`` と、公開しない ``inputs_private/``
（.gitignore 済み。guard の危険側の入力を置く）の両方から読み、ID の重複は誤りとする。

各行の形（共通で ``id``・``aim``（狙い）を持つ）::

    guard:     {"id", "aim", "input": "本文"} または "messages": [...],
               "expected": {"verdict": "安全"} / {"verdict": "有害", "categories": ["T8", ...]}
    embedding: {"id", "aim", "anchor": "基準の文", "near": ["近い文", ...], "far": ["遠い文", ...],
               "task_type": "SEMANTIC_SIMILARITY", "candidate_task_type": 省略可（既定は task_type）}
    rerank:    {"id", "aim", "query", "documents": [...], "expected_top": 0 始まりの位置,
               "expected_order": 省略可（全順位）, "task_type": 省略可, "query_instruction": 省略可}
    translate: {"id", "aim", "text", "source_lang": "ja"|"en", "target_lang": "en"|"ja"}
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sarashina_eval.models import (
    EMBEDDING_MODEL,
    EMBEDDING_TASK_TYPES,
    GUARD_MODEL,
    MODEL_RPM,
    RERANK_MODEL,
    RERANK_TASK_TYPES,
    TRANSLATE_MODEL,
)

#: 入力の種類と、送り先のモデル。
KIND_MODELS: dict[str, str] = {
    "guard": GUARD_MODEL,
    "embedding": EMBEDDING_MODEL,
    "rerank": RERANK_MODEL,
    "translate": TRANSLATE_MODEL,
}
#: guard の判定。
SAFE, HARMFUL = "安全", "有害"
#: guard のカテゴリ。
GUARD_CATEGORIES: frozenset[str] = frozenset(f"T{i}" for i in range(1, 13))
#: 固定 1 分窓で、窓が明けるまでに待つ秒数（rate_limiter に合わせ 61 秒で見積もる）。
WINDOW_SEC = 61


class CaseSetError(ValueError):
    """入力の組のファイルが壊れているときの例外。"""


@dataclass(frozen=True)
class Case:
    """入力の組の 1 行。

    Attributes
    ----------
    kind : str
        ``guard``・``embedding``・``rerank``・``translate``。
    id : str
        組の中で一意の識別子。
    aim : str
        狙い（人が読む説明）。
    data : dict
        行の残りの項目（形は種類ごと。モジュールの説明を参照）。
    source : str
        読んだファイルの置き場（``inputs`` か ``inputs_private`` など。記録用）。
    """

    kind: str
    id: str
    aim: str
    data: dict[str, Any]
    source: str = ""


@dataclass(frozen=True)
class PlannedCall:
    """送る予定の 1 要求。

    Attributes
    ----------
    kind : str
        入力の種類。
    model : str
        送り先のモデル（窓と済みの判定の鍵）。
    question_id : str
        記録の ``question_id``。embedding は ``<id>:anchor``・``<id>:near0``・``<id>:far0`` の形。
    method : str
        ``SarashinaClient`` のメソッド名。
    args : tuple
        位置引数。
    kwargs : dict
        キーワード引数（``question_id`` は含めない）。
    """

    kind: str
    model: str
    question_id: str
    method: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


def _require(cond: bool, where: str, msg: str) -> None:
    if not cond:
        raise CaseSetError(f"{where}: {msg}")


def _validate(kind: str, row: dict[str, Any], where: str) -> None:
    """種類ごとの必須項目と値を確かめる。"""
    if kind == "guard":
        _require(isinstance(row.get("input"), str) or isinstance(row.get("messages"), list),
                 where, "input（文字列）か messages（配列）が要る")
        exp = row.get("expected")
        _require(isinstance(exp, dict) and exp.get("verdict") in (SAFE, HARMFUL),
                 where, "expected.verdict は「安全」か「有害」")
        if exp["verdict"] == HARMFUL:
            cats = exp.get("categories")
            _require(isinstance(cats, list) and cats and set(cats) <= GUARD_CATEGORIES,
                     where, "有害なら expected.categories に T1〜T12 を 1 つ以上")
    elif kind == "embedding":
        _require(isinstance(row.get("anchor"), str), where, "anchor（文字列）が要る")
        for key in ("near", "far"):
            v = row.get(key)
            _require(isinstance(v, list) and v and all(isinstance(s, str) for s in v),
                     where, f"{key} は文字列の配列（1 件以上）")
        for key in ("task_type", "candidate_task_type"):
            if key in row or key == "task_type":
                _require(row.get(key) in EMBEDDING_TASK_TYPES, where, f"{key} が不正: {row.get(key)!r}")
    elif kind == "rerank":
        docs = row.get("documents")
        _require(isinstance(row.get("query"), str), where, "query（文字列）が要る")
        _require(isinstance(docs, list) and 1 <= len(docs) <= 20, where, "documents は 1〜20 件")
        top = row.get("expected_top")
        _require(isinstance(top, int) and 0 <= top < len(docs), where, "expected_top は documents の位置")
        order = row.get("expected_order")
        if order is not None:
            _require(sorted(order) == list(range(len(docs))), where, "expected_order は全文書の位置の並べ替え")
            _require(order[0] == top, where, "expected_order の先頭は expected_top と同じ")
        if "task_type" in row:
            _require(row["task_type"] in RERANK_TASK_TYPES, where, f"task_type が不正: {row['task_type']!r}")
    elif kind == "translate":
        _require(isinstance(row.get("text"), str), where, "text（文字列）が要る")
        pair = (row.get("source_lang"), row.get("target_lang"))
        _require(pair in (("ja", "en"), ("en", "ja")), where, "source_lang・target_lang は ja と en の組")


def load_cases(dirs: list[Path], kinds: list[str] | None = None) -> list[Case]:
    """置き場の並びから入力の組を読む。無い置き場・無いファイルは飛ばす。

    Parameters
    ----------
    dirs : list of Path
        ``inputs/``・``inputs_private/`` など。
    kinds : list of str, optional
        読む種類。省略時は全種類。

    Returns
    -------
    list of Case
        種類の順（``KIND_MODELS`` の順）、置き場の順、行の順。

    Raises
    ------
    CaseSetError
        JSON として読めない行、項目の不足・不正、同じ種類の中での ID の重複。
    """
    out: list[Case] = []
    seen: dict[tuple[str, str], str] = {}
    for kind in kinds or list(KIND_MODELS):
        if kind not in KIND_MODELS:
            raise CaseSetError(f"知らない種類: {kind}")
        for d in dirs:
            path = d / f"{kind}.jsonl"
            if not path.is_file():
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                where = f"{path}:{n}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise CaseSetError(f"{where}: JSON として読めない（{e}）") from e
                _require(isinstance(row, dict), where, "1 行は JSON のオブジェクト")
                cid, aim = row.get("id"), row.get("aim", "")
                _require(isinstance(cid, str) and cid and ":" not in cid, where, "id は ':' を含まない文字列")
                _require((kind, cid) not in seen, where, f"id {cid} が {seen.get((kind, cid))} と重複")
                _validate(kind, row, where)
                seen[(kind, cid)] = where
                data = {k: v for k, v in row.items() if k not in ("id", "aim")}
                out.append(Case(kind, cid, str(aim), data, d.name))
    return out


def embedding_ids(case: Case) -> list[tuple[str, str]]:
    """embedding の組から ``(役割, question_id)`` を並べる（anchor・near0..・far0..）。"""
    roles = ["anchor"] + [f"near{i}" for i in range(len(case.data["near"]))]
    roles += [f"far{i}" for i in range(len(case.data["far"]))]
    return [(r, f"{case.id}:{r}") for r in roles]


def plan_calls(cases: list[Case]) -> list[PlannedCall]:
    """入力の組から、送る要求を並べる。

    Parameters
    ----------
    cases : list of Case

    Returns
    -------
    list of PlannedCall
    """
    plan: list[PlannedCall] = []
    for c in cases:
        model, d = KIND_MODELS[c.kind], c.data
        if c.kind == "guard":
            msgs = d.get("messages") or [{"role": "user", "content": d["input"]}]
            plan.append(PlannedCall(c.kind, model, c.id, "chat", (model, msgs)))
        elif c.kind == "embedding":
            cand_tt = d.get("candidate_task_type", d["task_type"])
            for role, qid in embedding_ids(c):
                if role == "anchor":
                    text, tt = d["anchor"], d["task_type"]
                else:
                    key = "near" if role.startswith("near") else "far"
                    text, tt = d[key][int(role[len(key):])], cand_tt
                plan.append(PlannedCall(c.kind, model, qid, "embeddings", (text, tt)))
        elif c.kind == "rerank":
            kw = {k: d[k] for k in ("task_type", "query_instruction") if k in d}
            plan.append(PlannedCall(c.kind, model, c.id, "rerank", (d["query"], d["documents"]), kw))
        else:
            plan.append(PlannedCall(c.kind, model, c.id, "translate",
                                    (d["text"], d["source_lang"], d["target_lang"])))
    return plan


def send_call(client: Any, call: PlannedCall) -> dict[str, Any]:
    """1 要求を ``client`` のメソッドで送り、記録を返す。"""
    return getattr(client, call.method)(*call.args, question_id=call.question_id, **call.kwargs)


def estimate_seconds(plan: list[PlannedCall], rpm_by_model: dict[str, int] | None = None) -> dict[str, int]:
    """モデルごとの窓待ちの合計秒数を見積もる（応答の時間は含めない）。

    固定 1 分窓なので、``k`` 回送るモデルは ``ceil(k / rpm) - 1`` 回、窓の明けを待つ。
    """
    rpm = rpm_by_model or MODEL_RPM
    counts = Counter(c.model for c in plan)
    return {m: max(math.ceil(k / rpm[m]) - 1, 0) * WINDOW_SEC for m, k in counts.items()}


def format_dry_run(plan: list[PlannedCall], rpm_by_model: dict[str, int] | None = None) -> str:
    """``--dry-run`` で出す、要求数と所要時間の見積もりの文を作る（本文は出さない）。"""
    rpm = rpm_by_model or MODEL_RPM
    counts = Counter(c.model for c in plan)
    est = estimate_seconds(plan, rpm)
    lines = [f"送る要求 {len(plan)} 件（実際には送らない）:"]
    for i, c in enumerate(plan, start=1):
        lines.append(f"  {i:>3}. {c.model:<22} {c.question_id}")
    lines.append("所要時間の見積もり（窓待ちのみ。応答の時間は含めない）:")
    for m, sec in est.items():
        lines.append(f"  {m:<22} {counts[m]} 件 / {rpm[m]} RPM -> 約 {sec} 秒")
    lines.append(f"  全体（モデルを順に送るので合計） -> 約 {sum(est.values())} 秒")
    return "\n".join(lines)
