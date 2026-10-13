"""実行の一覧（logs/runs.jsonl）と、得点の CSV の名前を確かめる。"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from sarashina_eval import code_runner, score
from sarashina_eval.run_log import append_run, iter_records, run_date, run_log_files

RUN_ID = "abcdef0123456789"


def _rec(sent_at: str, run_id: str = RUN_ID, qid: str = "q1") -> dict:
    return {"run_id": run_id, "question_id": qid, "sent_at": sent_at,
            "request": {"body": {"model": "m"}},
            "response": {"status": 200, "body_json": {"choices": [{"message": {"content": "答え 3"}}]}}}


class RunRecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logs = Path(tempfile.mkdtemp())
        # 日をまたいだ実行: 最初の送信は 2026-10-11（UTC）。
        (self.logs / "2026-10-12.jsonl").write_text(
            json.dumps(_rec("2026-10-12T00:00:01.000+00:00", qid="q2")) + "\n"
            + json.dumps(_rec("2026-10-01T00:00:00.000+00:00", run_id="other")) + "\n", encoding="utf-8")
        (self.logs / "2026-10-11.jsonl").write_text(
            json.dumps(_rec("2026-10-11T23:59:59.000+00:00")) + "\n", encoding="utf-8")
        (self.logs / "2026-09-30.jsonl").write_text(
            json.dumps(_rec("2026-09-30T00:00:00.000+00:00", run_id="other")) + "\n", encoding="utf-8")

    def test_run_date_is_first_send_utc(self) -> None:
        self.assertEqual(run_date(self.logs, RUN_ID), date(2026, 10, 11))
        self.assertIsNone(run_date(self.logs, "nothing"))

    def test_run_log_files(self) -> None:
        self.assertEqual([p.name for p in run_log_files(self.logs, RUN_ID)],
                         ["2026-10-11.jsonl", "2026-10-12.jsonl"])

    def test_append_run_row_and_not_read_as_record(self) -> None:
        row = append_run(self.logs, event="send", run_id=RUN_ID, entry="sarashina_eval.run", note=None,
                         resumed=False, counts={"sent": 2}, outputs=[self.logs / "2026-10-11.jsonl"],
                         argv=["questions/"], now=datetime(2026, 10, 13, 1, 2, 3, 456789, timezone.utc))
        line = json.loads((self.logs / "runs.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(line, row)
        self.assertEqual(line["at"], "2026-10-13T01:02:03.456+00:00")
        self.assertEqual(list(line),
                         ["event", "run_id", "at", "entry", "note", "resumed", "counts", "outputs", "argv"])
        self.assertFalse(any(r.get("event") for r in iter_records(self.logs)))
        with self.assertRaises(ValueError):
            append_run(self.logs, event="x", run_id=RUN_ID, entry="e", note=None, resumed=False,
                       counts={}, outputs=[], argv=[])

    def test_score_main_names_csv_by_run_date_and_appends_runs(self) -> None:
        qdir = Path(tempfile.mkdtemp())
        rows = [{"id": i, "category": "math", "prompt": "p", "expected": "3",
                 "grading": {"method": "numeric"}, "intent": "i"} for i in ("q1", "q2")]
        (qdir / "q.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        reports = Path(tempfile.mkdtemp())
        argv = ["--run-id", RUN_ID, "--log-dir", str(self.logs), "--questions", str(qdir),
                "--reports-dir", str(reports), "--note", "再採点"]
        with redirect_stdout(io.StringIO()), mock.patch.object(code_runner, "docker_available", return_value=False):
            self.assertEqual(score.main(argv), 0)
            self.assertEqual(score.main(argv), 0)
        # 2 回採点しても同じ名前（上書き）で、採点した日ではなく最初の送信の日が入る。
        self.assertEqual([p.name for p in reports.iterdir()], ["2026-10-11-abcdef01-scores.csv"])
        runs = [json.loads(l) for l in (self.logs / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["note"], "再採点")
        self.assertEqual(runs[0]["counts"], {"answers": 2, "ungraded": 0, "correct": 2})
        self.assertEqual(runs[0]["argv"], argv)


if __name__ == "__main__":
    unittest.main()
