"""問題集を読み、各モデルへ流す実行の入口。

例::

    python3 -m sarashina_eval.run questions/ --dry-run
    python3 -m sarashina_eval.run questions/math.jsonl --models sarashina3-nano
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sarashina_eval.questions import CATEGORIES, QuestionSetError, load_questions
from sarashina_eval.runner import DEFAULT_MODELS, DEFAULT_N, format_dry_run, plan_requests, run_plan


def build_parser() -> argparse.ArgumentParser:
    """引数の解析器を作る。

    Returns
    -------
    argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(prog="python3 -m sarashina_eval.run", description="問題集を各モデルへ流す")
    p.add_argument("paths", nargs="+", type=Path, help="問題集の JSONL かそのディレクトリ")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="送り先のモデル")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="1 リクエストの候補数（1〜5）")
    p.add_argument("--category", choices=sorted(CATEGORIES), action="append", help="分野で絞る（複数可）")
    p.add_argument("--dry-run", action="store_true", help="送る要求と所要時間の見積もりを出し、送らない")
    p.add_argument("--log-dir", type=Path, default=None, help="記録の置き場（既定は logs/）")
    p.add_argument("--resume", metavar="RUN_ID", default=None,
                   help="この run_id の実行を再開し、HTTP 200 で済んだ (モデル, 設問) を送らない")
    return p


def main(argv: list[str] | None = None) -> int:
    """入口。

    Parameters
    ----------
    argv : list of str, optional
        引数。省略時は ``sys.argv[1:]``。

    Returns
    -------
    int
        終了コード。壊れた行や不正な指定は 2。
    """
    args = build_parser().parse_args(argv)
    try:
        questions = load_questions(args.paths)
        if args.category:
            questions = [q for q in questions if q.category in args.category]
        plan = plan_requests(questions, tuple(args.models), args.n)
    except QuestionSetError as e:
        print(e, file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"指定が不正: {e}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(format_dry_run(plan))
        return 0
    # 実際に送るときだけ client を作る（API キーを読むのはここだけ）。
    from sarashina_eval.client import SarashinaClient

    client = SarashinaClient(log_dir=args.log_dir, run_id=args.resume)
    records = run_plan(client, plan, should_skip=lambda model, qid: client.is_completed(model, qid))
    print(f"送信 {len(records)} 件、run_id={client.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
