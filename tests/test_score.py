"""自動採点（sarashina_eval.score）のテスト。ログは一時ディレクトリに書いた偽物を使う。"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from sarashina_eval import code_runner, score
from sarashina_eval.questions import Grading, Question


def q(qid: str, method: str, expected: str | None, category: str = "math", **params) -> Question:
    """テスト用の問題を作る。"""
    return Question(qid, category, "p", expected, Grading(method, params), "i")


def rec(run_id: str, model: str, qid: str, contents: list[str], status: int = 200) -> dict:
    """テスト用のログの 1 行を作る。"""
    return {
        "run_id": run_id,
        "question_id": qid,
        "attempt": 1,
        "request": {"body": {"model": model}},
        "response": {
            "status": status,
            "body_json": {"choices": [{"message": {"content": c}} for c in contents]} if status == 200 else None,
        },
    }


class ExtractNumberTest(unittest.TestCase):
    """実データの答えの形から数を取り出せること。"""

    def test_last_line_number(self) -> None:
        self.assertEqual(score.extract_number("3 × 14 = 42\n42 - 5 = 37\n\n37"), 37)

    def test_boxed_last(self) -> None:
        self.assertEqual(score.extract_number("途中で \\boxed{3} とおくと…答えは \\boxed{10}"), 10)

    def test_answer_with_display_math(self) -> None:
        self.assertEqual(score.extract_number("48 × 2.25 を計算する。\n答え： $$ 108 $$"), 108)

    def test_answer_with_inline_math(self) -> None:
        self.assertEqual(score.extract_number("3 × 14 = 42 なので、答えは $37$ です。"), 37)

    def test_bold_with_unit(self) -> None:
        self.assertEqual(score.extract_number("48 × 2.25 = **108** km"), 108)

    def test_fullwidth_and_thousands(self) -> None:
        self.assertEqual(score.extract_number("答えは１，２３４個"), 1234)

    def test_none(self) -> None:
        self.assertIsNone(score.extract_number("わかりません"))


class GradeAnswerTest(unittest.TestCase):
    """採点方法を問題ごとに選べること。"""

    def test_numeric_tolerance(self) -> None:
        self.assertEqual(score.grade_answer(q("a", "numeric", "10", tolerance=0.5), "10.4")[1], score.CORRECT)
        self.assertEqual(score.grade_answer(q("a", "numeric", "10"), "10.4")[1], score.WRONG)

    def test_numeric_extract_failed(self) -> None:
        self.assertEqual(score.grade_answer(q("a", "numeric", "10"), "不明"), (None, score.EXTRACT_FAILED))
        self.assertEqual(score.grade_answer(q("a", "numeric", "10"), None), (None, score.EXTRACT_FAILED))

    def test_exact_strips(self) -> None:
        self.assertEqual(score.grade_answer(q("a", "exact", "東京"), "  東京\n")[1], score.CORRECT)
        self.assertEqual(score.grade_answer(q("a", "exact", "東京"), "東京都")[1], score.WRONG)

    def test_regex_search(self) -> None:
        g = q("a", "regex", None, pattern=r"徳川(家康)?")
        self.assertEqual(score.grade_answer(g, "答えは徳川家康です"), ("徳川家康", score.CORRECT))
        self.assertEqual(score.grade_answer(g, "織田信長")[1], score.WRONG)

    def test_human_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score.grade_answer(q("a", "human", None), "x")


class ScoreRunTest(unittest.TestCase):
    """ログから採点し、CSV・JSONL を書くこと。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.logs = self.tmp / "logs"
        self.logs.mkdir()
        self.questions = {
            "math-001": q("math-001", "numeric", "37"),
            "math-002": q("math-002", "numeric", "108"),
            "math-003": q("math-003", "numeric", "10"),
            "code-001": q("code-001", "code_tests", None, category="code", function="f",
                          tests=[{"input": [], "output": 1}]),
        }
        lines = [
            rec("r1", "m-a", "math-001", ["x", "y", "z"]),  # 古い記録（後の 200 で置き換わる）
            rec("r1", "m-a", "math-001", ["42 - 5 = 37\n\n37", "答えは $37$ です。", "\\boxed{37}"]),
            rec("r1", "m-a", "math-001", [], status=429),  # 200 でないので使わない
            rec("r1", "m-a", "math-002", ["答え： $$ 108 $$", "**108** km", "100"]),
            rec("r1", "m-a", "math-003", ["9", "11", "不明"]),
            rec("r1", "m-a", "code-001", ["def f(): pass"] * 3),
            rec("r1", "m-b", "math-001", ["37"] * 3),
            rec("r2", "m-a", "math-003", ["10"] * 3),  # 別の run_id
        ]
        self.log_file = self.logs / "2026-10-13.jsonl"
        self.log_file.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")

    def test_last_ok_record_per_model_question(self) -> None:
        got = score.latest_ok_records(self.logs, "r1")
        self.assertEqual(score.answer_contents(got[("m-a", "math-001")])[2], "\\boxed{37}")
        self.assertNotIn(("m-a", "math-003-x"), got)
        self.assertEqual(len(got), 5)

    def test_each_of_n_answers_graded_and_bucketed(self) -> None:
        with mock.patch.object(code_runner, "docker_available", return_value=False):
            scores, ungraded, unknown = score.score_run(self.logs, "r1", self.questions)
        self.assertEqual(len(scores), 15)
        skipped = [s for s in scores if s.question_id == "code-001"]
        self.assertEqual({(s.status, s.note) for s in skipped}, {(score.SKIPPED, "docker_unavailable")})
        self.assertEqual(ungraded, [("m-a", "code", "code-001")])
        self.assertEqual(unknown, [])
        statuses = [s.status for s in scores if s.question_id == "math-003" and s.model == "m-a"]
        self.assertEqual(statuses, [score.WRONG, score.WRONG, score.EXTRACT_FAILED])
        header, rows = score.summarize(scores, ungraded)
        by = {r["category"]: r for r in rows}
        self.assertEqual(by["math"]["m-a:all_correct"], 1)
        self.assertEqual(by["math"]["m-a:partial"], 1)
        self.assertEqual(by["math"]["m-a:none_correct"], 1)
        self.assertEqual(by["math"]["m-a:accuracy"], "0.556")  # 5/9
        self.assertEqual(by["math"]["m-b:accuracy"], "1.000")
        self.assertEqual(by["code"]["m-a:ungraded"], 1)
        self.assertEqual(by["code"]["m-a:answers"], 0)  # 未採点は分母に入らない
        self.assertEqual(by["code"]["m-a:accuracy"], "")

    def test_main_writes_csv_and_jsonl_without_touching_log(self) -> None:
        before = self.log_file.read_bytes()
        qdir = self.tmp / "questions"
        qdir.mkdir()
        rows = [
            {"id": x.id, "category": x.category, "prompt": "p", "expected": x.expected,
             "grading": {"method": x.grading.method, **x.grading.params}, "intent": "i"}
            for x in self.questions.values()
        ]
        (qdir / "all.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        reports = self.tmp / "reports"
        out = io.StringIO()
        with redirect_stdout(out), mock.patch.object(code_runner, "docker_available", return_value=False):
            code = score.main(["--run-id", "r1", "--log-dir", str(self.logs), "--questions", str(qdir),
                               "--reports-dir", str(reports)])
        self.assertEqual(code, 0)
        self.assertIn("未採点", out.getvalue())
        self.assertIn("Docker が使えないため code_tests を飛ばした: 3 答え", out.getvalue())
        self.assertEqual(self.log_file.read_bytes(), before)
        csv_files = list(reports.glob("*-scores.csv"))
        self.assertEqual(len(csv_files), 1)
        text = csv_files[0].read_text(encoding="utf-8")
        self.assertNotIn("km", text)  # 出力本文を CSV に入れない
        table = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual([r["category"] for r in table], ["code", "math"])
        jl = [json.loads(l) for l in (self.logs / "scores" / "r1.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(jl), 15)
        self.assertEqual(set(jl[0]), {"run_id", "model", "question_id", "category", "method",
                                      "choice_index", "extracted", "status", "content",
                                      "tests_passed", "tests_total", "note"})
        self.assertEqual(sum(r["status"] == "skipped" for r in jl), 3)
        # 採点結果は logs/ 直下ではないので、ログの読み取りに混ざらない
        self.assertEqual(len(score.latest_ok_records(self.logs, "r1")), 5)


class CodeTestsScoringTest(unittest.TestCase):
    """code_tests を採点し、通った件数を採点結果に残すこと（Docker は偽物に差し替える）。"""

    def test_graded_with_counts(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        contents = ["```python\ndef f():\n    return 1\n```", "```python\ndef f():\n    return 2\n```", "コードなし"]
        (tmp / "l.jsonl").write_text(json.dumps(rec("r1", "m", "code-001", contents)) + "\n", encoding="utf-8")
        qs = {"code-001": q("code-001", "code_tests", None, category="code", function="f",
                            tests=[{"input": [], "output": 1}, {"input": [], "output": 1}])}

        def fake(code: str, params: dict) -> code_runner.CodeResult:
            return code_runner.CodeResult(2 if "return 1" in code else 0, 2)

        with mock.patch.object(code_runner, "docker_available", return_value=True), \
                mock.patch.object(code_runner, "run_tests", side_effect=fake):
            scores, ungraded, _ = score.score_run(tmp, "r1", qs)
        self.assertEqual(ungraded, [])
        self.assertEqual([(s.status, s.extracted, s.tests_passed, s.tests_total) for s in scores], [
            (score.CORRECT, "2/2", 2, 2), (score.WRONG, "0/2", 0, 2), (score.EXTRACT_FAILED, None, None, None)])
        _, rows = score.summarize(scores, ungraded)
        self.assertEqual(rows[0]["m:accuracy"], "0.333")


if __name__ == "__main__":
    unittest.main()
