"""Chat 以外の 4 モデル（guard・embedding・rerank・translate）を流して採点する入口。

例::

    python3 -m sarashina_eval.non_chat run --dry-run
    python3 -m sarashina_eval.non_chat run --kinds guard rerank
    python3 -m sarashina_eval.non_chat run --resume <run_id>
    python3 -m sarashina_eval.non_chat score --run-id <run_id>

入力は既定で ``inputs/`` と ``inputs_private/``（公開しない置き場）の両方から読む。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from sarashina_eval.non_chat_cases import KIND_MODELS, CaseSetError, format_dry_run, load_cases, plan_calls, send_call
from sarashina_eval.non_chat_score import score_cases, summary_lines, write_outputs
from sarashina_eval.run_log import append_run, run_date, run_log_files
from sarashina_eval.run_snapshot import save_snapshot, snapshot_paths

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIRS = (ROOT / "inputs", ROOT / "inputs_private")


def build_parser() -> argparse.ArgumentParser:
    """引数の解析器を作る。"""
    p = argparse.ArgumentParser(prog="python3 -m sarashina_eval.non_chat",
                                description="Chat 以外の 4 モデルを流して採点する")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "score"):
        s = sub.add_parser(name)
        s.add_argument("--inputs", nargs="+", type=Path, default=list(DEFAULT_INPUT_DIRS) if name == "run" else None,
                       help="入力の置き場（既定は inputs/ と inputs_private/。"
                            "score では既定で <log-dir>/runs/<run_id>/ の写しがあればそれ）")
        s.add_argument("--kinds", nargs="+", choices=list(KIND_MODELS), default=None, help="種類で絞る")
        s.add_argument("--log-dir", type=Path, default=ROOT / "logs")
        s.add_argument("--note", default=None, help="この実行・採点の目的（logs/runs.jsonl に残す）")
    r = sub.choices["run"]
    r.add_argument("--dry-run", action="store_true", help="リクエスト数と所要時間の見積もりを出し、送らない")
    r.add_argument("--resume", metavar="RUN_ID", default=None,
                   help="この run_id を再開し、HTTP 200 で済んだリクエストを送らない")
    s = sub.choices["score"]
    s.add_argument("--run-id", required=True)
    s.add_argument("--scores-dir", type=Path, default=None, help="既定は <log-dir>/scores")
    s.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return p


def main(argv: list[str] | None = None) -> int:
    """入口。

    Returns
    -------
    int
        終了コード。入力の誤りは 2、採点する記録が無ければ 1。
    """
    a = build_parser().parse_args(argv)
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    reference = None
    if a.command == "score" and a.inputs is None:
        snap = snapshot_paths(a.log_dir, a.run_id, "inputs")
        if snap is not None:
            a.inputs, reference = snap, {"source": "snapshot", "paths": [str(p) for p in snap]}
        else:
            a.inputs = list(DEFAULT_INPUT_DIRS)
            print(f"run_id {a.run_id} の写しが無いので、今の入力と照合する", file=sys.stderr)
    if a.command == "score" and reference is None:
        reference = {"source": "current", "paths": [str(p) for p in a.inputs]}
    try:
        cases = load_cases(a.inputs, a.kinds)
    except CaseSetError as e:
        print(e, file=sys.stderr)
        return 2
    if a.command == "run":
        plan = plan_calls(cases)
        if a.dry_run:
            print(format_dry_run(plan))
            return 0
        # 実際に送るときだけ client を作る（API キーを読むのはここだけ）。
        from sarashina_eval.client import SarashinaClient

        client = SarashinaClient(log_dir=a.log_dir, run_id=a.resume)
        # 送る前に、送る対象の入力の写しを残す（あとで同じ中身で採点し直すため）。
        by_kind: dict[str, set[str]] = {}
        for c in cases:
            by_kind.setdefault(c.kind, set()).add(c.id)
        snap = save_snapshot(a.log_dir, client.run_id, inputs=(a.inputs, by_kind))
        sent = 0
        for call in plan:
            if client.is_completed(call.model, call.question_id):
                continue
            send_call(client, call)
            sent += 1
        print(f"送信 {sent} 件、run_id={client.run_id}")
        append_run(a.log_dir, event="send", run_id=client.run_id, entry="sarashina_eval.non_chat run", note=a.note,
                   resumed=a.resume is not None, counts={"sent": sent},
                   outputs=[*run_log_files(a.log_dir, client.run_id), snap], argv=raw_argv)
        return 0
    scores = score_cases(a.log_dir, a.run_id, cases)
    if not scores:
        print("採点する入力が無い", file=sys.stderr)
        return 1
    jsonl, csv_path = write_outputs(scores, a.run_id, a.scores_dir or a.log_dir / "scores",
                                    a.results_dir, run_date(a.log_dir, a.run_id) or date.today())
    counts = {"cases": len(scores)}
    for s in scores:
        counts[s.status] = counts.get(s.status, 0) + 1
    append_run(a.log_dir, event="score", run_id=a.run_id, entry="sarashina_eval.non_chat score", note=a.note,
               resumed=False, counts=counts, outputs=[jsonl, csv_path], argv=raw_argv,
               reference=reference)
    print("\n".join(summary_lines(scores)))
    print(f"採点結果: {jsonl}\n得点: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
