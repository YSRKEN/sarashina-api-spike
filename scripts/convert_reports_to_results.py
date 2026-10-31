"""``reports/`` の古い形の得点の CSV を、``results/`` の新しい形へ一度だけ変換する。

採点の記録（``logs/``）が無いので採点は流し直さず、CSV の値だけを置き換える。

- 名前: ``<日付>-<実行 ID の全部>-<種類>-scores.csv``。種類の無い古い名前は Chat。
  実行 ID は、レポートに全部が書いてあるものだけ ``FULL_RUN_IDS`` で補い、ほかは先頭 8 文字のまま
- Chat: 横長（``<model>:<列>``）を、1 行が 1 モデル x 1 分野の縦長にする
- Chat 以外・Tool Calling: 先頭に ``run_id, run_date`` を足す
- 試しの実行（``TRIAL_RUN_IDS``）は ``results/trial/`` に置く

使い方::

    python3 scripts/convert_reports_to_results.py            # reports/ -> results/
    python3 scripts/convert_reports_to_results.py --src <旧> --dst <新>

古い CSV の削除はこのスクリプトでは行わない（``git rm`` で別に消す）。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sarashina_eval.results import CHAT_COLUMNS, scores_csv_path  # noqa: E402

#: 古い名前の中の実行 ID（先頭 8 文字や途中までの形）から、全部の実行 ID へ。
FULL_RUN_IDS = {"haiku55-": "haiku55-low-bcb885f0", "gpt6luna": "gpt6luna-low-8b0ad329"}

#: 試しの実行（本番ではない）。
TRIAL_RUN_IDS = frozenset({"8093aac1", "6587cdd3", "c04818ea"})

_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+?)(?:-(non-chat|tools))?-scores\.csv$")


def parse_name(name: str) -> tuple[date, str, str]:
    """古い CSV の名前を ``(日付, 実行 ID, 種類)`` に分ける。種類の無い名前は ``chat``。"""
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"想定外の名前: {name}")
    run_id = FULL_RUN_IDS.get(m.group(2), m.group(2))
    return date.fromisoformat(m.group(1)), run_id, m.group(3) or "chat"


def wide_to_long(header: list[str], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Chat の横長の行を縦長にする（モデルは見出しの順、分野は行の順）。"""
    models = list(dict.fromkeys(h.split(":")[0] for h in header[1:]))
    cols = CHAT_COLUMNS[4:]
    return [
        {"model": m, "category": r["category"], **{c: r[f"{m}:{c}"] for c in cols}}
        for m in models for r in rows
    ]


def convert(src: Path, dst_dir: Path) -> Path:
    """CSV 1 本を変換して書き、書いたパスを返す。"""
    day, run_id, kind = parse_name(src.name)
    out_dir = dst_dir / "trial" if run_id[:8] in TRIAL_RUN_IDS else dst_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header, rows = list(reader.fieldnames or []), list(reader)
    stamp = {"run_id": run_id, "run_date": f"{day:%Y-%m-%d}"}
    if kind == "chat":
        fields = list(CHAT_COLUMNS)
        out_rows = [{**stamp, **r} for r in wide_to_long(header, rows)]
    else:
        fields = ["run_id", "run_date", *header]
        out_rows = [{**stamp, **r} for r in rows]
    path = scores_csv_path(out_dir, day, run_id, kind)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    return path


def main(argv: list[str] | None = None) -> int:
    """入口。``--src`` の ``*-scores.csv`` を全部変換する。"""
    p = argparse.ArgumentParser(description="reports/ の CSV を results/ の新しい形へ変換する")
    p.add_argument("--src", type=Path, default=ROOT / "reports")
    p.add_argument("--dst", type=Path, default=ROOT / "results")
    a = p.parse_args(argv)
    for src in sorted(a.src.glob("*-scores.csv")):
        print(f"{src.name} -> {convert(src, a.dst).relative_to(a.dst)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
