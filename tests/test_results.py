"""得点の CSV の名前と、README の得点表（sarashina_eval.results・readme_table）のテスト。"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

from sarashina_eval import readme_table
from sarashina_eval.results import CHAT_COLUMNS, scores_csv_path

HEADER = ",".join(CHAT_COLUMNS)


def chat_csv(path: Path, run_id: str, model: str, cells: dict[str, tuple[str, int]]) -> None:
    """テスト用の縦長の Chat CSV を書く。``cells`` は ``{分野: (正解率, 答えの数)}``。"""
    lines = [HEADER]
    for cat, (acc, n) in cells.items():
        lines.append(f"{run_id},2026-10-13,{model},{cat},{acc},{n},0,0,0,{0 if n else 3}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CsvNameTest(unittest.TestCase):
    def test_name_uses_date_full_run_id_and_kind(self) -> None:
        p = scores_csv_path(Path("results"), date(2026, 10, 13), "haiku55-low-bcb885f0", "chat")
        self.assertEqual(p, Path("results") / "2026-10-13-haiku55-low-bcb885f0-chat-scores.csv")

    def test_unknown_kind_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scores_csv_path(Path("results"), date(2026, 10, 13), "r", "other")


class ReadmeTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.results = self.dir / "results"
        self.results.mkdir()
        rows = []
        for model, vals in (("sarashina3-mini", ("0.667", "0.767")), ("sarashina3-nano", ("1.000", "0.867"))):
            for cat, v in zip(("math", "logic"), vals):
                rows.append(f"main01,2026-10-13,{model},{cat},{v},30,0,0,0,0")
            rows.append(f"main01,2026-10-13,{model},japanese,,0,0,0,0,10")
        (self.results / "2026-10-13-main01-chat-scores.csv").write_text(
            HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        chat_csv(self.results / "2026-10-13-ref-low-ab12-chat-scores.csv", "ref-low-ab12", "other-model",
                 {"math": ("0.900", 10), "logic": ("0.800", 10), "japanese": ("", 0)})
        # 名前の末尾が似ていても、別の実行の CSV は拾わない。
        chat_csv(self.results / "2026-10-13-xmain01-chat-scores.csv", "xmain01", "bad", {"math": ("0.000", 1)})
        self.runs = [{"run_id": "ref-low-ab12", "heading": "参考"},
                     {"run_id": "main01", "heading": "2026-10-13"}]

    def test_reference_columns_go_to_the_right_end(self) -> None:
        lines = readme_table.build_table(self.results, self.runs).splitlines()
        self.assertEqual(
            lines[0], "| 分野 | sarashina3-mini（2026-10-13） | sarashina3-nano（2026-10-13） | other-model（参考） |")
        self.assertEqual(lines[1], "|---|---|---|---|")

    def test_rows_in_fixed_order_with_draft_for_zero_graded(self) -> None:
        lines = readme_table.build_table(self.results, self.runs).splitlines()
        self.assertEqual(lines[2], "| 数学 | 0.667 | 1.000 | 0.900 |")
        self.assertEqual(lines[3], "| 論理 | 0.767 | 0.867 | 0.800 |")
        self.assertEqual(lines[4], "| 日本語 | 下書きの判定 | 下書きの判定 | 下書きの判定 |")
        self.assertEqual(len(lines), 5)

    def test_missing_csv_is_an_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            readme_table.build_table(self.results, [{"run_id": "nothing", "heading": "x"}])

    def test_replace_between_markers_keeps_the_rest(self) -> None:
        text = "前\n<!-- results:start -->\n古い\n<!-- results:end -->\n後\n"
        out = readme_table.replace_block(text, "| 表 |")
        self.assertEqual(out, "前\n<!-- results:start -->\n| 表 |\n<!-- results:end -->\n後\n")
        self.assertEqual(readme_table.replace_block(out, "| 表 |"), out)  # 何度流しても同じ

    def test_markers_missing_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            readme_table.replace_block("印が無い\n", "| 表 |")

    def test_main_rewrites_readme(self) -> None:
        readme = self.dir / "README.md"
        readme.write_text("# 題\n<!-- results:start -->\n<!-- results:end -->\n", encoding="utf-8")
        (self.results / "readme-runs.json").write_text(
            json.dumps({"runs": self.runs}, ensure_ascii=False), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            code = readme_table.main(["--readme", str(readme), "--results-dir", str(self.results)])
        self.assertEqual(code, 0)
        text = readme.read_text(encoding="utf-8")
        self.assertIn("| 数学 | 0.667 | 1.000 | 0.900 |", text)
        self.assertTrue(text.startswith("# 題\n<!-- results:start -->\n| 分野 |"))


if __name__ == "__main__":
    unittest.main()
