"""Chat Completions の Tool Calling を、偽の道具とシナリオで試して採点する入口。

1 シナリオ × 1 モデル × 1 条件（``tool_choice`` の与え方）を「会話」と呼ぶ。会話は
モデルが ``tool_calls`` を返すたびに偽の道具（``fake_tools``）を実行し、結果を ``role: "tool"`` の
メッセージで返して送り直す。往復の上限を超えたら打ち切る。

各往復は ``SarashinaClient.chat`` で送るので、窓・送り直し・ログの記録は既存のまま効く。
記録の ``question_id`` は ``<シナリオ id>#<条件>``、``attempt`` は往復の番号（1 から）。
最後の往復の記録の ``request.body.messages`` と応答を足せば、会話の全体が組み立て直せる。

使い方::

    python3 -m sarashina_eval.tools run tool_scenarios/ --dry-run
    python3 -m sarashina_eval.tools run tool_scenarios/ --models sarashina3-nano --tool-choice omit auto
    python3 -m sarashina_eval.tools score --run-id <id>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from sarashina_eval import fake_tools
from sarashina_eval.client_options import timeout_kwargs
from sarashina_eval.models import MODEL_RPM, REASONING_EFFORTS, reasoning_params
from sarashina_eval.results import scores_csv_path
from sarashina_eval.run_log import append_run, iter_records, run_date, run_log_files
from sarashina_eval.run_snapshot import question_files, save_snapshot, snapshot_paths
from sarashina_eval.score import normalize_text

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS: tuple[str, ...] = ("sarashina3-mini", "sarashina3-nano")
#: 往復の上限の既定。
DEFAULT_MAX_ROUNDS = 5

#: ``tool_choice`` の条件。値は往復の番号（1 から）を受けて、送る ``tool_choice`` を返す（None は省略）。
CONDITIONS: dict[str, Any] = {
    "omit": lambda rnd: None,
    "auto": lambda rnd: "auto",
    "required": lambda rnd: "required",
    "required-first": lambda rnd: "required" if rnd == 1 else "auto",
}
DEFAULT_CONDITIONS: tuple[str, ...] = ("omit", "auto")
#: ``question_id`` の中でシナリオ id と条件を分ける文字。
SEP = "#"


class ScenarioError(ValueError):
    """シナリオのファイルが壊れている。"""


@dataclass(frozen=True)
class Scenario:
    """1 件のシナリオ。

    Attributes
    ----------
    id : str
    prompt : str
        ユーザーの発話。
    tools : list of str
        渡す道具の名前（``fake_tools.TOOL_DEFS`` の鍵）。
    expected_calls : list of dict
        期待する呼び出し ``{"name": ..., "args": {...}}``。空なら「呼ばない」ことを期待する。
    expected_facts : list of str
        最後の答えに含まれるべき文字列（正規化して部分一致）。
    aim : str
        狙い。
    system : str or None
        system メッセージ（任意）。
    """

    id: str
    prompt: str
    tools: list[str]
    expected_calls: list[dict[str, Any]]
    expected_facts: list[str]
    aim: str = ""
    system: str | None = None

    def messages(self) -> list[dict[str, Any]]:
        """最初に送る ``messages`` を返す。"""
        head = [{"role": "system", "content": self.system}] if self.system else []
        return [*head, {"role": "user", "content": self.prompt}]


def load_scenarios(paths: list[Path]) -> list[Scenario]:
    """シナリオの JSONL（またはそのディレクトリ）を読む。

    Parameters
    ----------
    paths : list of Path

    Returns
    -------
    list of Scenario

    Raises
    ------
    ScenarioError
        読めない行、項目の不足、知らない道具、id の重複のとき。
    """
    out: list[Scenario] = []
    seen: set[str] = set()
    for f in question_files(paths):
        for no, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{f}:{no}"
            try:
                obj = json.loads(line)
                sc = Scenario(
                    id=obj["id"], prompt=obj["prompt"], tools=list(obj["tools"]),
                    expected_calls=list(obj.get("expected_calls", [])),
                    expected_facts=list(obj.get("expected_facts", [])),
                    aim=obj.get("aim", ""), system=obj.get("system"),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ScenarioError(f"{where}: シナリオとして読めない: {e}") from e
            unknown = [t for t in sc.tools if t not in fake_tools.TOOL_DEFS]
            if unknown:
                raise ScenarioError(f"{where}: 知らない道具 {unknown}")
            if SEP in sc.id or sc.id in seen:
                raise ScenarioError(f"{where}: id が重複しているか {SEP!r} を含む: {sc.id}")
            seen.add(sc.id)
            out.append(sc)
    return out


class ChatSender(Protocol):
    """``SarashinaClient.chat`` と同じ形で送れるもの。"""

    def chat(self, model: str, messages: list[dict[str, Any]], *, question_id: str | None = None,
             attempt: int = 1, **params: Any) -> dict[str, Any]: ...


def _message_of(record: dict[str, Any]) -> dict[str, Any] | None:
    """記録から ``choices[0].message`` を取り出す。無ければ None。"""
    body = (record.get("response") or {}).get("body_json")
    try:
        msg = body["choices"][0]["message"]
    except (TypeError, KeyError, IndexError):
        return None
    return msg if isinstance(msg, dict) else None


def run_conversation(
    client: ChatSender,
    model: str,
    scenario: Scenario,
    condition: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    reasoning_effort: str | None = None,
) -> list[dict[str, Any]]:
    """1 会話を往復させる。

    Parameters
    ----------
    client : ChatSender
    model : str
    scenario : Scenario
    condition : str
        ``CONDITIONS`` の鍵。
    max_rounds : int
        送る回数の上限。上限の回でもまだ ``tool_calls`` が来たら打ち切る。
    reasoning_effort : str, optional
        推論の強さ（思考モデルだけ）。毎回の送信に付ける。

    Returns
    -------
    list of dict
        往復ごとの記録。
    """
    choose = CONDITIONS[condition]
    messages = scenario.messages()
    tools = fake_tools.tool_definitions(scenario.tools)
    extra = reasoning_params([model], reasoning_effort)
    records = []
    for rnd in range(1, max_rounds + 1):
        params: dict[str, Any] = {"tools": tools, **extra}
        tc = choose(rnd)
        if tc is not None:
            params["tool_choice"] = tc
        rec = client.chat(model, list(messages), question_id=f"{scenario.id}{SEP}{condition}", attempt=rnd,
                          **params)
        records.append(rec)
        msg = _message_of(rec)
        if (rec.get("response") or {}).get("status") != 200 or msg is None:
            break
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        # 公式の手順どおり、応答の message をそのまま足し、呼び出しごとに tool メッセージを足す。
        messages.append(msg)
        for call in calls:
            fn = call.get("function") or {}
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "content": fake_tools.execute(str(fn.get("name")), fn.get("arguments") or "")})
    return records


def plan(
    scenarios: list[Scenario], models: list[str], conditions: list[str], reasoning_effort: str | None = None
) -> list[tuple[str, Scenario, str]]:
    """送る会話を ``(model, scenario, condition)`` で並べる。

    Raises
    ------
    ValueError
        知らないモデル・条件か、推論の強さを受け付けないモデルのとき。
    """
    reasoning_params(models, reasoning_effort)
    for m in models:
        if m not in MODEL_RPM or m == "sarashina3-guard":
            raise ValueError(f"tools を使えないか知らないモデル: {m}")
    for c in conditions:
        if c not in CONDITIONS:
            raise ValueError(f"知らない条件: {c}（{', '.join(CONDITIONS)}）")
    return [(m, s, c) for m in models for c in conditions for s in scenarios]


def format_dry_run(items: list[tuple[str, Scenario, str]], max_rounds: int) -> str:
    """送る予定と、送信回数の最小・最大を文字列にする。"""
    lines = [f"会話 {len(items)} 件（往復の上限 {max_rounds}）"]
    by_model: dict[str, int] = {}
    for m, s, c in items:
        lines.append(f"  {m}  {s.id}{SEP}{c}  tools={','.join(s.tools)}")
        by_model[m] = by_model.get(m, 0) + 1
    for m, k in by_model.items():
        lo, hi = k, k * max_rounds
        lines.append(f"{m}: 送信 {lo}〜{hi} 回（{MODEL_RPM[m]} 回/分。最大なら窓待ち約 {max(-(-hi // MODEL_RPM[m]) - 1, 0)} 分）")
    return "\n".join(lines)


def latest_records(log_dir: Path, run_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    """``run_id`` の記録のうち、``(model, question_id)`` ごとに最後に送ったものを返す。

    再開で会話をやり直すと、後から送った記録が会話の最新になる。
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in iter_records(log_dir):
        if rec.get("run_id") != run_id:
            continue
        model = ((rec.get("request") or {}).get("body") or {}).get("model")
        qid = rec.get("question_id")
        if not isinstance(model, str) or not isinstance(qid, str) or SEP not in qid:
            continue
        key = (model, qid)
        prev = out.get(key)
        if prev is None or (rec.get("sent_at") or "") >= (prev.get("sent_at") or ""):
            out[key] = rec
    return out


def is_finished(rec: dict[str, Any], max_rounds: int) -> bool:
    """会話の最後の記録が、答えで終わったか上限で打ち切ったかを返す（再開で飛ばす判定）。"""
    msg = _message_of(rec)
    if (rec.get("response") or {}).get("status") != 200 or msg is None:
        return False
    return not msg.get("tool_calls") or int(rec.get("attempt", 1)) >= max_rounds


# ---- 採点 ----


@dataclass
class ToolScore:
    """1 会話の採点。``passed`` は呼び出し・引数・事実・打ち切りなしのすべてを満たすとき真。"""

    run_id: str
    model: str
    scenario_id: str
    condition: str
    status: str
    rounds: int
    truncated: bool
    calls: list[dict[str, Any]] = field(default_factory=list)
    args_json_ok: bool = True
    called_expected: bool = False
    args_match: bool = False
    unexpected_calls: int = 0
    facts_found: list[str] = field(default_factory=list)
    facts_missing: list[str] = field(default_factory=list)
    final_answer: str | None = None
    passed: bool = False


def _norm(v: Any) -> str:
    return normalize_text(str(v)).strip().lower()


def _args_cover(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """期待した引数がすべて、実際の引数の値に（正規化して）含まれるか。"""
    return all(k in actual and _norm(v) in _norm(actual[k]) for k, v in expected.items())


def score_conversation(rec: dict[str, Any], scenario: Scenario, condition: str) -> ToolScore:
    """会話の最後の記録から会話を組み立て直して採点する。

    Parameters
    ----------
    rec : dict
        会話の最後の往復の記録。
    scenario : Scenario
    condition : str

    Returns
    -------
    ToolScore
    """
    model = rec["request"]["body"]["model"]
    status = (rec.get("response") or {}).get("status")
    msg = _message_of(rec)
    sent = rec["request"]["body"].get("messages", [])
    assistants = [m for m in sent if m.get("role") == "assistant"] + ([msg] if msg else [])
    calls: list[dict[str, Any]] = []
    json_ok = True
    for m in assistants:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                args, json_ok = None, False
            calls.append({"name": fn.get("name"), "args": args})
    truncated = bool(msg and msg.get("tool_calls"))
    final = None if (msg is None or truncated) else msg.get("content")
    s = ToolScore(run_id=rec["run_id"], model=model, scenario_id=scenario.id, condition=condition,
                  status="ok" if status == 200 and msg is not None else f"http_{status}",
                  rounds=int(rec.get("attempt", 1)), truncated=truncated, calls=calls, args_json_ok=json_ok,
                  final_answer=final)
    exp = scenario.expected_calls
    names = [c["name"] for c in calls]
    if exp:
        s.called_expected = all(e["name"] in names for e in exp)
        s.args_match = all(any(c["name"] == e["name"] and isinstance(c["args"], dict)
                               and _args_cover(c["args"], e.get("args", {})) for c in calls) for e in exp)
        s.unexpected_calls = sum(n not in {e["name"] for e in exp} for n in names)
    else:
        s.called_expected = s.args_match = not calls
        s.unexpected_calls = len(calls)
    text = _norm(final or "")
    s.facts_found = [f for f in scenario.expected_facts if _norm(f) in text]
    s.facts_missing = [f for f in scenario.expected_facts if _norm(f) not in text]
    s.passed = (s.status == "ok" and not truncated and json_ok and s.called_expected and s.args_match
                and not s.facts_missing)
    return s


CSV_HEADER = ["run_id", "run_date", "model", "scenario_id", "condition", "status", "passed", "rounds", "truncated", "n_calls",
              "called_expected", "args_json_ok", "args_match", "unexpected_calls", "facts_found", "facts_total"]


def csv_row(s: ToolScore, day: date) -> dict[str, Any]:
    """本文を含まない CSV の 1 行にする。``day`` は実行の最初の送信の UTC 日付。"""
    return {"run_id": s.run_id, "run_date": f"{day:%Y-%m-%d}", "model": s.model, "scenario_id": s.scenario_id, "condition": s.condition, "status": s.status,
            "passed": int(s.passed), "rounds": s.rounds, "truncated": int(s.truncated), "n_calls": len(s.calls),
            "called_expected": int(s.called_expected), "args_json_ok": int(s.args_json_ok),
            "args_match": int(s.args_match), "unexpected_calls": s.unexpected_calls,
            "facts_found": len(s.facts_found), "facts_total": len(s.facts_found) + len(s.facts_missing)}


def score_run(log_dir: Path, run_id: str, scenarios: dict[str, Scenario]) -> tuple[list[ToolScore], list[str]]:
    """``run_id`` の会話をすべて採点する。

    Returns
    -------
    tuple
        ``(採点の並び, シナリオに無い id)``。
    """
    scores, unknown = [], []
    for (model, qid), rec in sorted(latest_records(log_dir, run_id).items()):
        sid, cond = qid.split(SEP, 1)
        if sid not in scenarios:
            unknown.append(sid)
            continue
        scores.append(score_conversation(rec, scenarios[sid], cond))
    return scores, unknown


def write_outputs(scores: list[ToolScore], run_id: str, scores_dir: Path, results_dir: Path,
                  day: date) -> tuple[Path, Path]:
    """本文つきの JSONL（``<scores_dir>/<run_id>-tools.jsonl``）と本文なしの CSV を書く。"""
    scores_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl = scores_dir / f"{run_id}-tools.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    csv_path = scores_csv_path(results_dir, day, run_id, "tools")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(csv_row(s, day) for s in scores)
    return jsonl, csv_path


# ---- 入口 ----


def build_parser() -> argparse.ArgumentParser:
    """引数の解析器を作る。"""
    p = argparse.ArgumentParser(prog="python3 -m sarashina_eval.tools", description="Tool Calling を試して採点する")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="シナリオを各モデルへ流す")
    r.add_argument("paths", nargs="*", type=Path, default=[ROOT / "tool_scenarios"], help="シナリオの JSONL かそのディレクトリ")
    r.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    r.add_argument("--tool-choice", nargs="+", default=list(DEFAULT_CONDITIONS), choices=list(CONDITIONS),
                   help="条件。omit は tool_choice を送らない（API の既定 required）。required-first は 1 回目だけ required")
    r.add_argument("--reasoning-effort", choices=list(REASONING_EFFORTS), default=None,
                   help="推論の強さ（思考モデルだけ）。省略時は送らない")
    r.add_argument("--timeout", type=float, default=None, help="1 リクエストの時間切れの秒数（既定 60）")
    r.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS, help="1 会話の送信の上限")
    r.add_argument("--dry-run", action="store_true", help="送る予定を出し、送らない")
    r.add_argument("--log-dir", type=Path, default=None)
    r.add_argument("--resume", metavar="RUN_ID", default=None, help="この run_id の、終わった会話を送らない")
    r.add_argument("--note", default=None)
    s = sub.add_parser("score", help="ログを採点する")
    s.add_argument("--run-id", required=True)
    s.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    s.add_argument("--scenarios", type=Path, default=None, help="既定は <log-dir>/runs/<run_id>/ の写し")
    s.add_argument("--scores-dir", type=Path, default=None)
    s.add_argument("--results-dir", type=Path, default=ROOT / "results")
    s.add_argument("--note", default=None)
    return p


def _run(a: argparse.Namespace, argv: list[str]) -> int:
    try:
        scenarios = load_scenarios(a.paths)
        items = plan(scenarios, a.models, a.tool_choice, a.reasoning_effort)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if a.max_rounds < 1:
        print("--max-rounds は 1 以上", file=sys.stderr)
        return 2
    if a.dry_run:
        print(format_dry_run(items, a.max_rounds))
        return 0
    from sarashina_eval.client import SarashinaClient  # 実際に送るときだけ（API キーを読むのはここだけ）

    client = SarashinaClient(log_dir=a.log_dir, run_id=a.resume, **timeout_kwargs(a.timeout))
    log_dir = a.log_dir if a.log_dir is not None else ROOT / "logs"
    snap = save_snapshot(log_dir, client.run_id, questions=(a.paths, {s.id for s in scenarios}))
    done = {k for k, rec in latest_records(log_dir, client.run_id).items() if is_finished(rec, a.max_rounds)}
    sent = conv = 0
    for model, sc, cond in items:
        if (model, f"{sc.id}{SEP}{cond}") in done:
            continue
        sent += len(run_conversation(client, model, sc, cond, a.max_rounds, a.reasoning_effort))
        conv += 1
    print(f"会話 {conv} 件、送信 {sent} 件、run_id={client.run_id}")
    append_run(log_dir, event="send", run_id=client.run_id, entry="sarashina_eval.tools", note=a.note,
               resumed=a.resume is not None, counts={"sent": sent, "conversations": conv},
               outputs=[*run_log_files(log_dir, client.run_id), snap], argv=argv)
    return 0


def _score(a: argparse.Namespace, argv: list[str]) -> int:
    if a.scenarios is not None:
        paths, reference = [a.scenarios], {"source": "current", "paths": [str(a.scenarios)]}
    else:
        snap = snapshot_paths(a.log_dir, a.run_id, "questions")
        if snap is None:
            paths = [ROOT / "tool_scenarios"]
            reference = {"source": "current", "paths": [str(paths[0])]}
            print(f"run_id {a.run_id} の写しが無いので、今の tool_scenarios/ と照合する", file=sys.stderr)
        else:
            paths, reference = snap, {"source": "snapshot", "paths": [str(p) for p in snap]}
    scores, unknown = score_run(a.log_dir, a.run_id, {s.id: s for s in load_scenarios(paths)})
    if not scores:
        print(f"run_id {a.run_id} の会話の記録が無い", file=sys.stderr)
        return 1
    day = run_date(a.log_dir, a.run_id) or date.today()
    jsonl, csv_path = write_outputs(scores, a.run_id, a.scores_dir or a.log_dir / "scores", a.results_dir, day)
    counts = {"conversations": len(scores), "passed": sum(s.passed for s in scores),
              "truncated": sum(s.truncated for s in scores)}
    append_run(a.log_dir, event="score", run_id=a.run_id, entry="sarashina_eval.tools", note=a.note,
               resumed=False, counts=counts, outputs=[jsonl, csv_path], argv=argv, reference=reference)
    print(f"採点結果: {jsonl}\n得点: {csv_path}\n通過 {counts['passed']}/{len(scores)}、打ち切り {counts['truncated']}")
    if unknown:
        print(f"シナリオに無い id（飛ばした）: {', '.join(sorted(set(unknown)))}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """入口。

    Returns
    -------
    int
        終了コード。不正な指定は 2、採点する記録が無ければ 1。
    """
    args_list = sys.argv[1:] if argv is None else argv
    a = build_parser().parse_args(args_list)
    return _run(a, args_list) if a.cmd == "run" else _score(a, args_list)


if __name__ == "__main__":
    sys.exit(main())
