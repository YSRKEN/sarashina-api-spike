"""コード分野の採点（sarashina_eval.code_runner）のテスト。

Docker を使うテストは、Docker が使えるときだけ走らせる（使えなければ skip）。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from sarashina_eval import code_runner, score
from sarashina_eval.questions import load_questions

ROOT = Path(__file__).resolve().parent.parent
SOLUTIONS = Path(__file__).resolve().parent / "code_solutions"
DOCKER = code_runner.docker_available()

ADD = {"function": "add", "tests": [{"input": [1, 2], "output": 3}, {"input": [-1, 1], "output": 0}]}


def block(code: str) -> str:
    """コードをコードブロックに包む。"""
    return f"説明\n```python\n{code}\n```\n"


class ExtractCodeTest(unittest.TestCase):
    """答えの本文からコードブロックを抜き出せること。"""

    def test_python_block(self) -> None:
        self.assertEqual(code_runner.extract_code(block("def f():\n    return 1")), "def f():\n    return 1")

    def test_plain_fence_and_multiple(self) -> None:
        got = code_runner.extract_code("```\nimport math\n```\nそして\n```py\nx = 1\n```")
        self.assertEqual(got, "import math\n\nx = 1")

    def test_other_language_ignored(self) -> None:
        self.assertIsNone(code_runner.extract_code("```bash\nls\n```"))

    def test_no_block(self) -> None:
        self.assertIsNone(code_runner.extract_code("def f(): return 1"))
        self.assertIsNone(code_runner.extract_code(None))


class ValidateParamsTest(unittest.TestCase):
    """テストの形を検査すること。"""

    def test_ok(self) -> None:
        code_runner.validate_params(ADD)

    def test_bad(self) -> None:
        for bad in ({}, {"function": "1x", "tests": [{"input": [], "output": 0}]}, {"function": "f", "tests": []},
                    {"function": "f", "tests": [{"input": 1, "output": 0}]}):
            with self.assertRaises(ValueError):
                code_runner.validate_params(bad)


class QuestionSetTest(unittest.TestCase):
    """questions/code.jsonl が読めて、テストの形が正しいこと。"""

    def test_code_questions(self) -> None:
        qs = load_questions([ROOT / "questions" / "code.jsonl"])
        self.assertEqual(len(qs), 10)
        for q in qs:
            self.assertEqual((q.category, q.grading.method), ("code", "code_tests"))
            code_runner.validate_params(q.grading.params)
            self.assertIn(q.grading.params["function"], q.prompt)
            self.assertIn("コードブロック", q.prompt)


@unittest.skipUnless(DOCKER, "Docker が使えない")
class DockerRunTest(unittest.TestCase):
    """Docker の上で実行し、制限が効くこと。"""

    def run_code(self, code: str, timeout: float = code_runner.TIMEOUT_SEC) -> code_runner.CodeResult:
        return code_runner.run_tests(code, ADD, timeout=timeout)

    def test_correct_all_pass(self) -> None:
        self.assertEqual(self.run_code("def add(a, b):\n    print('noise')\n    return a + b"),
                         code_runner.CodeResult(2, 2, None))

    def test_partial(self) -> None:
        res = self.run_code("def add(a, b):\n    return 3")
        self.assertEqual((res.passed, res.total, res.all_passed), (1, 2, False))

    def test_infinite_loop_stops(self) -> None:
        res = self.run_code("def add(a, b):\n    while True:\n        pass", timeout=5)
        self.assertEqual(res.passed, 0)
        self.assertTrue(res.error.startswith("timeout"), res.error)

    def test_network_fails(self) -> None:
        code = ("import urllib.request\n"
                "def add(a, b):\n"
                "    urllib.request.urlopen('http://example.com', timeout=3)\n"
                "    return a + b")
        res = self.run_code(code)
        self.assertEqual(res.passed, 0)

    def test_file_write_fails(self) -> None:
        for path in ("/tmp/x", "/x", "/root/x"):
            code = f"def add(a, b):\n    open({path!r}, 'w').write('x')\n    return a + b"
            self.assertEqual(self.run_code(code).passed, 0, path)

    def test_syntax_error(self) -> None:
        res = self.run_code("def add(a, b) return")
        self.assertEqual(res.passed, 0)
        self.assertIn("SyntaxError", res.error)

    def test_fake_result_line_not_trusted(self) -> None:
        code = ("import sys\n"
                "sys.__stdout__.write('__RESULT_x__' + '{\"passed\": 2, \"total\": 2}\\n')\n"
                "def add(a, b):\n    return 99")
        self.assertEqual(self.run_code(code).passed, 0)

    @unittest.skipUnless(SOLUTIONS.is_dir(), "模範解答（tests/code_solutions）が無い")
    def test_reference_solutions_pass(self) -> None:
        for q in load_questions([ROOT / "questions" / "code.jsonl"]):
            sol = (SOLUTIONS / f"{q.id.replace('-', '_')}.py").read_text(encoding="utf-8")
            extracted, status, res = score.grade_code_answer(q, block(sol))
            with self.subTest(q.id):
                self.assertEqual(status, score.CORRECT, res)
                print(f"{q.id}: {extracted}")


if __name__ == "__main__":
    unittest.main()
