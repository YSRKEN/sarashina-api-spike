"""一度きりの変換スクリプト（scripts/convert_reports_to_results.py）のテスト。"""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from datetime import date
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "convert_reports_to_results", Path(__file__).resolve().parent.parent / "scripts" / "convert_reports_to_results.py")
conv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(conv)

WIDE = ("category,m-a:accuracy,m-a:answers,m-a:all_correct,m-a:partial,m-a:none_correct,m-a:ungraded,"
        "m-b:accuracy,m-b:answers,m-b:all_correct,m-b:partial,m-b:none_correct,m-b:ungraded\n"
        "math,0.667,30,6,2,2,0,1.000,30,10,0,0,0\n"
        "japanese,,0,0,0,0,10,,0,0,0,0,10\n")


class ConvertTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.src, self.dst = self.dir / "reports", self.dir / "results"
        self.src.mkdir()

    def _rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def test_parse_name(self) -> None:
        d = date(2026, 10, 13)
        self.assertEqual(conv.parse_name("2026-10-13-31fb3e0c-scores.csv"), (d, "31fb3e0c", "chat"))
        self.assertEqual(conv.parse_name("2026-10-13-339a0d86-non-chat-scores.csv"), (d, "339a0d86", "non-chat"))
        self.assertEqual(conv.parse_name("2026-10-13-450389c8-tools-scores.csv"), (d, "450389c8", "tools"))
        self.assertEqual(conv.parse_name("2026-10-13-haiku55--scores.csv"), (d, "haiku55-low-bcb885f0", "chat"))
        self.assertEqual(conv.parse_name("2026-10-13-gpt6luna-scores.csv")[1], "gpt6luna-low-8b0ad329")

    def test_chat_becomes_long_one_row_per_model_and_category(self) -> None:
        (self.src / "2026-10-13-31fb3e0c-scores.csv").write_text(WIDE, encoding="utf-8")
        out = conv.convert(self.src / "2026-10-13-31fb3e0c-scores.csv", self.dst)
        self.assertEqual(out, self.dst / "2026-10-13-31fb3e0c-chat-scores.csv")
        rows = self._rows(out)
        self.assertEqual([(r["model"], r["category"], r["accuracy"], r["ungraded"]) for r in rows],
                         [("m-a", "math", "0.667", "0"), ("m-a", "japanese", "", "10"),
                          ("m-b", "math", "1.000", "0"), ("m-b", "japanese", "", "10")])
        self.assertEqual({(r["run_id"], r["run_date"]) for r in rows}, {("31fb3e0c", "2026-10-13")})

    def test_other_kinds_get_run_columns_in_front(self) -> None:
        (self.src / "2026-10-13-450389c8-tools-scores.csv").write_text("model,status\nm,ok\n", encoding="utf-8")
        out = conv.convert(self.src / "2026-10-13-450389c8-tools-scores.csv", self.dst)
        self.assertEqual(self._rows(out), [{"run_id": "450389c8", "run_date": "2026-10-13", "model": "m", "status": "ok"}])

    def test_trial_runs_go_to_trial_folder(self) -> None:
        (self.src / "2026-10-13-8093aac1-scores.csv").write_text(WIDE, encoding="utf-8")
        out = conv.convert(self.src / "2026-10-13-8093aac1-scores.csv", self.dst)
        self.assertEqual(out.parent, self.dst / "trial")


if __name__ == "__main__":
    unittest.main()
