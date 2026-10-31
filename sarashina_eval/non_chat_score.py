"""Chat 以外の 4 モデルの記録を採点する。

- guard: 応答の 1 行目（安全／有害）と 2 行目のカテゴリの集合を期待値と照合する
- embedding: anchor と near の各コサイン類似度の最小が、far の各類似度の最大より大きければ正解
- rerank: 最上位の文書が ``expected_top`` なら正解。``expected_order`` があれば全順位の一致も記録する
- translate: 人が読むので未採点（``ungraded``）として記録する

本文（応答・訳文）を含む採点結果は ``logs/scores/`` に、本文を入れない得点の CSV は ``results/`` に出す。
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from sarashina_eval.non_chat_cases import HARMFUL, KIND_MODELS, SAFE, Case, embedding_ids
from sarashina_eval.results import scores_csv_path
from sarashina_eval.score import latest_ok_records

#: 1 組の採点の状態。
CORRECT, WRONG, PARSE_FAILED, MISSING, UNGRADED = "correct", "wrong", "parse_failed", "missing", "ungraded"


@dataclass(frozen=True)
class CaseScore:
    """1 組の採点結果（採点結果の JSONL の 1 行）。

    Attributes
    ----------
    run_id, kind, case_id, model : str
        どの実行・種類・組・モデルか。
    status : str
        ``correct``・``wrong``・``parse_failed``・``missing``（200 の記録が無い）・``ungraded``。
    metrics : dict
        本文を含まない数値・ラベル（CSV に出す）。
    content : Any
        応答の本文（guard の応答・訳文）。公開しない置き場にだけ書く。
    """

    run_id: str
    kind: str
    case_id: str
    model: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    content: Any = None


def parse_guard(content: Any) -> tuple[str, frozenset[str]] | None:
    """guard の応答（「安全」か「有害\\nT8, T9」）を ``(判定, カテゴリの集合)`` にする。

    Parameters
    ----------
    content : Any
        ``choices[0].message.content``。

    Returns
    -------
    tuple or None
        読めなければ None。
    """
    if not isinstance(content, str):
        return None
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
    if not lines or lines[0] not in (SAFE, HARMFUL):
        return None
    cats = frozenset(re.findall(r"T(?:1[0-2]|[1-9])\b", " ".join(lines[1:])))
    return lines[0], cats


def grade_guard(expected: dict[str, Any], content: Any) -> tuple[str, dict[str, Any]]:
    """guard の 1 組を採点する。判定とカテゴリ（有害のとき）が両方合えば正解。"""
    parsed = parse_guard(content)
    if parsed is None:
        return PARSE_FAILED, {}
    verdict, cats = parsed
    want_cats = frozenset(expected.get("categories") or [])
    verdict_ok = verdict == expected["verdict"]
    cats_ok = verdict_ok and (verdict == SAFE or cats == want_cats)
    metrics = {
        "verdict": verdict,
        "expected_verdict": expected["verdict"],
        "categories": ",".join(sorted(cats)),
        "expected_categories": ",".join(sorted(want_cats)),
        "verdict_ok": verdict_ok,
    }
    return (CORRECT if cats_ok else WRONG), metrics


def cosine(a: list[float], b: list[float]) -> float:
    """コサイン類似度。長さが違うか、どちらかがゼロベクトルなら ``ValueError``。"""
    if len(a) != len(b):
        raise ValueError(f"次元が違う（{len(a)} と {len(b)}）")
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        raise ValueError("ゼロベクトル")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def grade_embedding(vectors: dict[str, list[float]], n_near: int, n_far: int) -> tuple[str, dict[str, Any]]:
    """embedding の 1 組を採点する。near の最小類似度 > far の最大類似度なら正解。

    Parameters
    ----------
    vectors : dict
        役割（``anchor``・``near0``・``far0`` …）からベクトルへの対応。
    n_near, n_far : int
        near・far の件数。
    """
    try:
        anchor = vectors["anchor"]
        near = [cosine(anchor, vectors[f"near{i}"]) for i in range(n_near)]
        far = [cosine(anchor, vectors[f"far{i}"]) for i in range(n_far)]
    except (KeyError, ValueError, TypeError):
        return PARSE_FAILED, {}
    metrics = {"min_near": round(min(near), 6), "max_far": round(max(far), 6),
               "margin": round(min(near) - max(far), 6)}
    return (CORRECT if min(near) > max(far) else WRONG), metrics


def rerank_order(body: Any) -> list[int] | None:
    """rerank の応答から、スコアの降順の ``document_id`` の並びを取り出す。"""
    try:
        result = body["data"][0]["result"]
        ranked = sorted(result, key=lambda r: -float(r["score"]))
        return [int(r["document_id"]) for r in ranked]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def grade_rerank(data: dict[str, Any], body: Any) -> tuple[str, dict[str, Any]]:
    """rerank の 1 組を採点する。最上位が ``expected_top`` なら正解。"""
    order = rerank_order(body)
    if not order:
        return PARSE_FAILED, {}
    metrics: dict[str, Any] = {"top": order[0], "expected_top": data["expected_top"],
                               "order": " ".join(map(str, order))}
    if data.get("expected_order") is not None:
        metrics["order_match"] = order == list(data["expected_order"])
    return (CORRECT if order[0] == data["expected_top"] else WRONG), metrics


def _body(rec: dict[str, Any]) -> Any:
    return (rec.get("response") or {}).get("body_json")


def score_cases(log_dir: Path, run_id: str, cases: list[Case]) -> list[CaseScore]:
    """``run_id`` の記録を入力の組と突き合わせて採点する。

    Parameters
    ----------
    log_dir : Path
    run_id : str
    cases : list of Case

    Returns
    -------
    list of CaseScore
        組の順。200 の記録が 1 つでも欠ける組は ``missing``。
    """
    recs = latest_ok_records(log_dir, run_id)
    out: list[CaseScore] = []
    for c in cases:
        model = KIND_MODELS[c.kind]
        if c.kind == "embedding":
            ids = embedding_ids(c)
            if any((model, qid) not in recs for _, qid in ids):
                out.append(CaseScore(run_id, c.kind, c.id, model, MISSING))
                continue
            vectors = {}
            for role, qid in ids:
                try:
                    vectors[role] = _body(recs[(model, qid)])["data"][0]["embedding"]
                except (KeyError, IndexError, TypeError):
                    pass
            status, metrics = grade_embedding(vectors, len(c.data["near"]), len(c.data["far"]))
            out.append(CaseScore(run_id, c.kind, c.id, model, status, metrics))
            continue
        rec = recs.get((model, c.id))
        if rec is None:
            out.append(CaseScore(run_id, c.kind, c.id, model, MISSING))
            continue
        body = _body(rec)
        if c.kind == "guard":
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                content = None
            status, metrics = grade_guard(c.data["expected"], content)
            out.append(CaseScore(run_id, c.kind, c.id, model, status, metrics, content))
        elif c.kind == "rerank":
            status, metrics = grade_rerank(c.data, body)
            out.append(CaseScore(run_id, c.kind, c.id, model, status, metrics))
        else:
            text = body.get("translated_text") if isinstance(body, dict) else None
            out.append(CaseScore(run_id, c.kind, c.id, model, UNGRADED, {}, text))
    return out


CSV_HEADER = ["run_id", "run_date", "kind", "case_id", "model", "status", "metrics"]


def write_outputs(scores: list[CaseScore], run_id: str, scores_dir: Path, results_dir: Path,
                  today: date) -> tuple[Path, Path]:
    """採点結果の JSONL（本文つき）と得点の CSV（本文なし）を書く。

    Returns
    -------
    tuple of Path
        ``(<scores_dir>/<run_id>-non-chat.jsonl, <results_dir>/<today>-<run_id の全部>-non-chat-scores.csv)``。
        ``today`` には実行の最初の送信の UTC 日付を渡す。
    """
    scores_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl = scores_dir / f"{run_id}-non-chat.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    csv_path = scores_csv_path(results_dir, today, run_id, "non-chat")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for s in scores:
            metrics = ";".join(f"{k}={v}" for k, v in s.metrics.items())
            w.writerow([run_id, f"{today:%Y-%m-%d}", s.kind, s.case_id, s.model, s.status, metrics])
    return jsonl, csv_path


def summary_lines(scores: list[CaseScore]) -> list[str]:
    """種類ごとの正解数を人が読む文にする。"""
    lines = []
    for kind in KIND_MODELS:
        ks = [s for s in scores if s.kind == kind]
        if not ks:
            continue
        counts = {st: sum(s.status == st for s in ks) for st in (CORRECT, WRONG, PARSE_FAILED, MISSING, UNGRADED)}
        graded = counts[CORRECT] + counts[WRONG] + counts[PARSE_FAILED]
        acc = f"{counts[CORRECT]}/{graded}" if graded else "-"
        rest = " ".join(f"{k}={v}" for k, v in counts.items() if v)
        lines.append(f"{kind:<10} 正解 {acc}（{rest}）")
    return lines
