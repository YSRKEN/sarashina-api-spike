"""README の印の間に、分野ごとの得点表を書き込む。

載せる実行は ``results/readme-runs.json`` だけで決める::

    {"runs": [{"run_id": "31fb3e0c", "heading": "2026-10-13"},
              {"run_id": "haiku55-low-bcb885f0", "heading": "参考"}]}

実行ごとに ``results/*-<run_id>-chat-scores.csv`` を読み、モデルごとに 1 列にする。
見出しに「参考」を含む実行（ほかの会社のモデル）は右端へ寄せる。新しい版の実行を一覧に足すと列が増える。

使い方::

    python3 -m sarashina_eval.readme_table [--readme README.md]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

START, END = "<!-- results:start -->", "<!-- results:end -->"

#: 表に出す分野の順と表示名。
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("math", "数学"), ("logic", "論理"), ("instruction", "指示追従"),
    ("code", "コード"), ("japanese", "日本語"), ("knowledge", "知識"),
)

#: 採点した答えが 0 の分野に出す語（数は出さない）。
DRAFT = "下書きの判定"


def load_runs(path: Path) -> list[dict[str, str]]:
    """``readme-runs.json`` の ``runs``（``run_id`` と ``heading`` の並び）を読む。"""
    return json.loads(path.read_text(encoding="utf-8"))["runs"]


def _read_run(results_dir: Path, run_id: str) -> list[dict[str, str]]:
    """``run_id`` の Chat の CSV の行を返す（名前の末尾で照合し、中の ``run_id`` でも絞る）。

    Raises
    ------
    FileNotFoundError
        CSV が無いとき。
    """
    found = sorted(results_dir.glob(f"*-{run_id}-chat-scores.csv"))
    found = [p for p in found if p.name[len("0000-00-00-"):] == f"{run_id}-chat-scores.csv"]
    if not found:
        raise FileNotFoundError(f"実行 {run_id} の Chat の CSV が {results_dir} に無い")
    with found[0].open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["run_id"] == run_id]


def build_table(results_dir: Path, runs: list[dict[str, str]]) -> str:
    """分野ごとの得点表（Markdown）を作る。

    列は一覧の順（「参考」の実行は右端へ寄せ、その中は一覧の順）に、モデルごとに 1 列。
    採点した答えが 0 の分野は ``下書きの判定``。その列に行が無い分野は ``-``。
    """
    ordered = sorted(runs, key=lambda r: "参考" in r["heading"])  # 安定な並べ替え
    columns: list[tuple[str, dict[str, dict[str, str]]]] = []
    for run in ordered:
        by_model: dict[str, dict[str, dict[str, str]]] = {}
        for row in _read_run(results_dir, run["run_id"]):
            by_model.setdefault(row["model"], {})[row["category"]] = row
        for model, cats in by_model.items():
            columns.append((f"{model}（{run['heading']}）", cats))
    lines = ["| 分野 | " + " | ".join(h for h, _ in columns) + " |",
             "|---" * (len(columns) + 1) + "|"]
    present = {c for _, cats in columns for c in cats}
    for key, label in CATEGORIES:
        if key not in present:
            continue
        cells = []
        for _, cats in columns:
            row = cats.get(key)
            cells.append("-" if row is None else row["accuracy"] if int(row["answers"]) > 0 else DRAFT)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def replace_block(text: str, table: str) -> str:
    """``START`` と ``END`` の間を ``table`` に置き換える。印の外は触らない。

    Raises
    ------
    ValueError
        印が無い、または順序が逆のとき。
    """
    s, e = text.find(START), text.find(END)
    if s < 0 or e < s:
        raise ValueError(f"{START} と {END} の印が、この順で README に必要")
    return text[:s + len(START)] + "\n" + table + "\n" + text[e:]


def main(argv: list[str] | None = None) -> int:
    """入口。README の印の間を書き換える。"""
    p = argparse.ArgumentParser(prog="python3 -m sarashina_eval.readme_table",
                                description="README の印の間に分野ごとの得点表を書く")
    p.add_argument("--readme", type=Path, default=ROOT / "README.md")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--runs", type=Path, default=None, help="既定は <results-dir>/readme-runs.json")
    a = p.parse_args(argv)
    runs = load_runs(a.runs or a.results_dir / "readme-runs.json")
    table = build_table(a.results_dir, runs)
    a.readme.write_text(replace_block(a.readme.read_text(encoding="utf-8"), table), encoding="utf-8")
    print(f"書き換えた: {a.readme}\n{table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
