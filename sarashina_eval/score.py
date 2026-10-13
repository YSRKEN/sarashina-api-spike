"""ログから答えを取り出して自動採点し、分野×モデルの得点を CSV に出す。

ログ（``logs/*.jsonl``）は読むだけで書き換えない。答えごとの採点結果は
``logs/scores/<run_id>.jsonl`` に、集計は ``reports/<日付>-<run_id の先頭 8 文字>-scores.csv`` に出す。
日付はその実行の最初の送信（``sent_at``）の UTC 日付。採点のたびに ``logs/runs.jsonl`` へ 1 行足す。

使い方::

    python3 -m sarashina_eval.score --run-id <id>

採点できる方法は exact・numeric・regex・code_tests。code_tests は答えのコードブロックを Docker の
``python:3`` で実行し、全テストが通れば正解とする（通った件数も採点結果に残す）。Docker が使えないときは
code_tests を飛ばし、状態 ``skipped`` の行を採点結果に書いて「未採点」として数える。human は採点しない
（「未採点」として数え、正解率の分母に入れない）。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from sarashina_eval import code_runner
from sarashina_eval.questions import Question, load_questions
from sarashina_eval.run_log import append_run, run_date
from sarashina_eval.run_snapshot import snapshot_paths
from sarashina_eval.run_log import iter_records

ROOT = Path(__file__).resolve().parent.parent

#: 自動で採点する方法。
AUTO_METHODS: frozenset[str] = frozenset({"exact", "numeric", "regex", "code_tests"})

#: 答え 1 つの採点の状態。
CORRECT, WRONG, EXTRACT_FAILED = "correct", "wrong", "extract_failed"

#: 採点を飛ばした答えの状態（Docker が使えない code_tests）。正解率の分母に入れない。
SKIPPED = "skipped"

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class AnswerScore:
    """答え 1 つの採点結果（採点結果の JSONL の 1 行）。

    Attributes
    ----------
    run_id, model, question_id, category, method : str
        どの実行・モデル・問題・分野・採点方法か。
    choice_index : int
        ``choices`` の中の位置。
    extracted : str or None
        取り出した値。取り出せなければ None。
    status : str
        ``correct``・``wrong``・``extract_failed``・``skipped`` のいずれか。
    content : str or None
        答えの本文。公開しない置き場にだけ書く。
    tests_passed, tests_total : int or None
        code_tests で通ったテストの件数と全件数。それ以外の方法と、実行しなかったときは None。
    note : str or None
        補足（code_tests の実行の失敗の理由、飛ばした理由）。
    """

    run_id: str
    model: str
    question_id: str
    category: str
    method: str
    choice_index: int
    extracted: str | None
    status: str
    content: str | None
    tests_passed: int | None = None
    tests_total: int | None = None
    note: str | None = None


def normalize_text(text: str) -> str:
    """数を読めるように本文を正規化する。

    全角を半角に（NFKC）し、LaTeX の ``$``・強調の ``**`` を除き、数字の間の桁区切りのカンマを除く。

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    t = unicodedata.normalize("NFKC", text).replace("$", "").replace("**", "")
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)


def extract_number(text: str) -> float | None:
    """本文から答えの数を取り出す。

    順に、(1) 最後の ``\\boxed{...}`` の中身の数、(2) 最後に出る「答え」の後ろで最初の数、
    (3) 本文で最後に出る数。

    Parameters
    ----------
    text : str

    Returns
    -------
    float or None
        取り出せなければ None。
    """
    t = normalize_text(text)
    boxed = _BOXED.findall(t)
    if boxed:
        m = _NUMBER.search(boxed[-1])
        return float(m.group()) if m else None
    idx = t.rfind("答え")
    while idx != -1:
        m = _NUMBER.search(t, idx)
        if m:
            return float(m.group())
        idx = t.rfind("答え", 0, idx)
    nums = _NUMBER.findall(t)
    return float(nums[-1]) if nums else None


def grade_answer(question: Question, content: str | None) -> tuple[str | None, str]:
    """答え 1 つを採点する。

    Parameters
    ----------
    question : Question
        ``grading.method`` が ``AUTO_METHODS`` のいずれかの問題。
    content : str or None
        答えの本文。

    Returns
    -------
    tuple
        ``(取り出した値, 状態)``。状態は ``correct``・``wrong``・``extract_failed``。

    Raises
    ------
    ValueError
        自動で採点しない方法のとき。
    """
    method, params = question.grading.method, question.grading.params
    if method not in AUTO_METHODS:
        raise ValueError(f"採点方法 {method!r} は自動で採点しない")
    if not isinstance(content, str):
        return None, EXTRACT_FAILED
    if method == "exact":
        got = content.strip()
        return got, CORRECT if got == (question.expected or "").strip() else WRONG
    if method == "code_tests":
        extracted, status, _ = grade_code_answer(question, content)
        return extracted, status
    if method == "regex":
        m = re.search(params["pattern"], content)
        return (m.group(), CORRECT) if m else (None, WRONG)
    value = extract_number(content)
    if value is None:
        return None, EXTRACT_FAILED
    tol = float(params.get("tolerance", 0))
    ok = abs(value - float(question.expected or "nan")) <= tol
    return f"{value:g}", CORRECT if ok else WRONG


def grade_code_answer(
    question: Question, content: str | None
) -> tuple[str | None, str, code_runner.CodeResult | None]:
    """code_tests の答え 1 つを Docker で実行して採点する。

    Parameters
    ----------
    question : Question
        ``grading.method`` が ``code_tests`` の問題。
    content : str or None

    Returns
    -------
    tuple
        ``(通った件数 "k/n", 状態, 実行結果)``。コードブロックが無ければ
        ``(None, extract_failed, None)``。
    """
    code = code_runner.extract_code(content)
    if code is None:
        return None, EXTRACT_FAILED, None
    res = code_runner.run_tests(code, question.grading.params)
    return f"{res.passed}/{res.total}", CORRECT if res.all_passed else WRONG, res


def latest_ok_records(log_dir: Path, run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    """``run_id`` の記録から、``(model, question_id)`` ごとに HTTP 200 の最後の記録を選ぶ。

    Parameters
    ----------
    log_dir : Path
    run_id : str

    Returns
    -------
    dict
        ``(model, question_id)`` から記録への対応。
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in iter_records(log_dir):
        if rec.get("run_id") != run_id or (rec.get("response") or {}).get("status") != 200:
            continue
        model = ((rec.get("request") or {}).get("body") or {}).get("model")
        qid = rec.get("question_id")
        if isinstance(model, str) and isinstance(qid, str):
            out[(model, qid)] = rec
    return out


def answer_contents(rec: dict[str, Any]) -> list[str | None]:
    """記録の ``response.body_json.choices[].message.content`` を並べる。"""
    choices = ((rec.get("response") or {}).get("body_json") or {}).get("choices") or []
    return [((c or {}).get("message") or {}).get("content") for c in choices]


def score_run(
    log_dir: Path, run_id: str, questions: dict[str, Question]
) -> tuple[list[AnswerScore], list[tuple[str, str, str]], list[str]]:
    """1 回の実行を採点する。

    Parameters
    ----------
    log_dir : Path
    run_id : str
    questions : dict
        ID から問題への対応。

    Returns
    -------
    tuple
        ``(答えごとの採点結果, 未採点の (model, category, question_id), 問題集に無い question_id)``。
    """
    scores: list[AnswerScore] = []
    ungraded: list[tuple[str, str, str]] = []
    unknown: list[str] = []
    docker_ok: bool | None = None
    for (model, qid), rec in sorted(latest_ok_records(log_dir, run_id).items()):
        q = questions.get(qid)
        if q is None:
            unknown.append(qid)
            continue
        if q.grading.method not in AUTO_METHODS:
            ungraded.append((model, q.category, qid))
            continue
        if q.grading.method == "code_tests":
            if docker_ok is None:
                docker_ok = code_runner.docker_available()
            for i, content in enumerate(answer_contents(rec)):
                if not docker_ok:
                    scores.append(AnswerScore(run_id, model, qid, q.category, q.grading.method, i, None, SKIPPED,
                                              content, note="docker_unavailable"))
                    continue
                extracted, status, res = grade_code_answer(q, content)
                scores.append(AnswerScore(
                    run_id, model, qid, q.category, q.grading.method, i, extracted, status, content,
                    res.passed if res else None, res.total if res else None, res.error if res else None,
                ))
            if not docker_ok:
                ungraded.append((model, q.category, qid))
            continue
        for i, content in enumerate(answer_contents(rec)):
            extracted, status = grade_answer(q, content)
            scores.append(AnswerScore(run_id, model, qid, q.category, q.grading.method, i, extracted, status, content))
    return scores, ungraded, unknown


def summarize(
    scores: list[AnswerScore], ungraded: list[tuple[str, str, str]]
) -> tuple[list[str], list[dict[str, Any]]]:
    """分野×モデルに集計し、CSV の見出しと行を作る。

    問題ごとに、全問正解（例 3/3）・一部・全滅（0/3）に分ける。正解率は採点した答えのうち正解の割合。

    Parameters
    ----------
    scores : list of AnswerScore
    ungraded : list of tuple
        ``(model, category, question_id)``。

    Returns
    -------
    tuple
        ``(見出し, 行の並び)``。行は分野、列はモデルごとに
        ``<model>:accuracy``・``:answers``・``:all_correct``・``:partial``・``:none_correct``・``:ungraded``。
    """
    per_q: dict[tuple[str, str, str], list[str]] = {}
    for s in scores:
        if s.status == SKIPPED:
            continue
        per_q.setdefault((s.category, s.model, s.question_id), []).append(s.status)
    models = sorted({s.model for s in scores if s.status != SKIPPED} | {u[0] for u in ungraded})
    cats = sorted({s.category for s in scores if s.status != SKIPPED} | {u[1] for u in ungraded})
    stats: dict[tuple[str, str], dict[str, int]] = {}

    def cell(c: str, m: str) -> dict[str, int]:
        return stats.setdefault((c, m), dict.fromkeys(("correct", "answers", "all", "partial", "none", "ungraded"), 0))

    for (c, m, _), sts in per_q.items():
        st, k = cell(c, m), sts.count(CORRECT)
        st["correct"] += k
        st["answers"] += len(sts)
        st["all" if k == len(sts) and k > 0 else "partial" if k > 0 else "none"] += 1
    for m, c, _ in ungraded:
        cell(c, m)["ungraded"] += 1

    suffixes = ("accuracy", "answers", "all_correct", "partial", "none_correct", "ungraded")
    header = ["category"] + [f"{m}:{s}" for m in models for s in suffixes]
    rows = []
    for c in cats:
        row: dict[str, Any] = {"category": c}
        for m in models:
            st = cell(c, m)
            acc = f"{st['correct'] / st['answers']:.3f}" if st["answers"] else ""
            vals = (acc, st["answers"], st["all"], st["partial"], st["none"], st["ungraded"])
            row.update({f"{m}:{s}": v for s, v in zip(suffixes, vals)})
        rows.append(row)
    return header, rows


def write_outputs(
    scores: list[AnswerScore],
    header: list[str],
    rows: list[dict[str, Any]],
    run_id: str,
    scores_dir: Path,
    reports_dir: Path,
    today: date,
) -> tuple[Path, Path]:
    """採点結果の JSONL と得点の CSV を書く。

    CSV の名前は ``<today>-<run_id の先頭 8 文字>-scores.csv``。``today`` には実行の最初の送信の
    UTC 日付を渡す（同じ実行を採点し直すと同じ名前で上書きし、別の実行とは重ならない）。

    JSONL は答えの本文を含むので、公開しない置き場（既定 ``logs/scores/``）に置く。
    CSV には本文を入れない。

    Returns
    -------
    tuple of Path
        ``(JSONL のパス, CSV のパス)``。
    """
    scores_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    jsonl = scores_dir / f"{run_id}.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    csv_path = reports_dir / f"{today:%Y-%m-%d}-{run_id[:8]}-scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return jsonl, csv_path


def main(argv: list[str] | None = None) -> int:
    """入口。``--run-id`` の実行を採点して書き出す。"""
    p = argparse.ArgumentParser(prog="python3 -m sarashina_eval.score", description="ログを自動採点する")
    p.add_argument("--run-id", required=True)
    p.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    p.add_argument("--questions", type=Path, default=None,
                   help="照合する問題集（既定は <log-dir>/runs/<run_id>/ の写し、無ければ questions/）")
    p.add_argument("--scores-dir", type=Path, default=None, help="既定は <log-dir>/scores")
    p.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    p.add_argument("--note", default=None, help="この採点の目的（logs/runs.jsonl に残す）")
    a = p.parse_args(argv)
    if a.questions is not None:
        qpaths = [a.questions]
        reference = {"source": "current", "paths": [str(a.questions)]}
    else:
        snap = snapshot_paths(a.log_dir, a.run_id, "questions")
        if snap is not None:
            qpaths, reference = snap, {"source": "snapshot", "paths": [str(p) for p in snap]}
        else:
            qpaths = [ROOT / "questions"]
            reference = {"source": "current", "paths": [str(qpaths[0])]}
            print(f"run_id {a.run_id} の写しが無いので、今の questions/ と照合する", file=sys.stderr)
    questions = {q.id: q for q in load_questions(qpaths)}
    scores, ungraded, unknown = score_run(a.log_dir, a.run_id, questions)
    if not scores and not ungraded:
        print(f"run_id {a.run_id} の HTTP 200 の記録が無い", file=sys.stderr)
        return 1
    header, rows = summarize(scores, ungraded)
    day = run_date(a.log_dir, a.run_id) or date.today()
    jsonl, csv_path = write_outputs(
        scores, header, rows, a.run_id, a.scores_dir or a.log_dir / "scores", a.reports_dir, day
    )
    counts = {"answers": len(scores), "ungraded": len(ungraded)}
    for s in scores:
        counts[s.status] = counts.get(s.status, 0) + 1
    append_run(a.log_dir, event="score", run_id=a.run_id, entry="sarashina_eval.score", note=a.note,
               resumed=False, counts=counts, outputs=[jsonl, csv_path],
               argv=sys.argv[1:] if argv is None else argv, reference=reference)
    print(f"採点結果: {jsonl}\n得点: {csv_path}")
    skipped = sum(s.status == SKIPPED for s in scores)
    if skipped:
        print(f"Docker が使えないため code_tests を飛ばした: {skipped} 答え（採点結果に status=skipped で記録）")
    if ungraded:
        print(f"未採点（Docker の無い code_tests・human。正解率の分母に入れない）: {len(ungraded)} 件")
    failed = sum(s.status == EXTRACT_FAILED for s in scores)
    if failed:
        print(f"取り出し失敗（不正解として数えた）: {failed} 件")
    if unknown:
        print(f"問題集に無い ID（飛ばした）: {', '.join(sorted(set(unknown)))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
