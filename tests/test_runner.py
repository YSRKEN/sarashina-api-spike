"""問題集の読み込み・検査と、偽のサーバーへの送信・記録・dry-run を確かめる。"""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from sarashina_eval import SarashinaClient
from sarashina_eval import run as run_cli
from sarashina_eval.questions import QuestionSetError, load_questions
from sarashina_eval.runner import estimate_seconds, format_dry_run, plan_requests, run_plan

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "sample_math.jsonl"
FAKE_KEY = "test-key-not-real"
START = 1_800_000_000.0  # 2027-01-15 08:00:00 UTC


class FakeClock:
    """sleep で時刻を進めるだけの偽の時計。"""

    def __init__(self, start: float) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class _Handler(BaseHTTPRequestHandler):
    """受けた本文を記録し、n 個の候補を返す。"""

    received: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.received.append(body)
        choices = [{"index": i, "message": {"role": "assistant", "content": str(i)}} for i in range(body.get("n", 1))]
        payload = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


VALID = {
    "id": "x-1",
    "category": "logic",
    "prompt": "問い",
    "expected": "答え",
    "grading": {"method": "exact"},
    "intent": "狙い",
}


class LoadQuestionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, name: str, lines: list[str]) -> Path:
        p = self.dir / name
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def test_sample_loads(self) -> None:
        qs = load_questions([SAMPLE])
        self.assertEqual([q.id for q in qs], ["math-001", "math-002", "math-003"])
        self.assertEqual(qs[0].grading.method, "numeric")
        self.assertEqual(qs[0].grading.params, {"tolerance": 0})

    def test_extension_fields_kept(self) -> None:
        row = dict(VALID, category="code", system="簡潔に",
                   grading={"method": "code_tests", "tests": [{"input": "1", "output": "2"}]}, tags=["t"])
        q = load_questions([self._write("code.jsonl", [json.dumps(row, ensure_ascii=False)])])[0]
        self.assertEqual(q.grading.params["tests"], [{"input": "1", "output": "2"}])
        self.assertEqual(q.messages()[0], {"role": "system", "content": "簡潔に"})

    def test_broken_rows_rejected_with_line_numbers(self) -> None:
        no_prompt = {k: v for k, v in VALID.items() if k != "prompt"}
        lines = [
            json.dumps(VALID, ensure_ascii=False),
            "{これは JSON でない",
            "",
            json.dumps(no_prompt, ensure_ascii=False),
            json.dumps(dict(VALID, id="x-2", category="history"), ensure_ascii=False),
            json.dumps(dict(VALID, id="x-3", grading={"method": "vibes"}), ensure_ascii=False),
            json.dumps(VALID, ensure_ascii=False),
        ]
        path = self._write("logic.jsonl", lines)
        with self.assertRaises(QuestionSetError) as cm:
            load_questions([path])
        probs = cm.exception.problems
        self.assertEqual(len(probs), 5)
        self.assertTrue(probs[0].startswith(f"{path}:2: JSON"))
        self.assertIn(f"{path}:4: 必須の項目 'prompt'", probs[1])
        self.assertIn(f"{path}:5: 分野 'history'", probs[2])
        self.assertIn(f"{path}:6: 採点方法 'vibes'", probs[3])
        self.assertIn(f"{path}:7: ID 'x-1' が", probs[4])

    def test_directory_reads_all_jsonl(self) -> None:
        self._write("a.jsonl", [json.dumps(VALID, ensure_ascii=False)])
        self._write("b.jsonl", [json.dumps(dict(VALID, id="x-2"), ensure_ascii=False)])
        self.assertEqual([q.id for q in load_questions([self.dir])], ["x-1", "x-2"])


class PlanTest(unittest.TestCase):
    def test_plan_both_models_n3_without_temperature(self) -> None:
        plan = plan_requests(load_questions([SAMPLE]))
        self.assertEqual(len(plan), 6)
        self.assertEqual({r.model for r in plan}, {"sarashina3-mini", "sarashina3-nano"})
        for r in plan:
            self.assertEqual(r.params, {"n": 3})

    def test_invalid_n_and_model(self) -> None:
        qs = load_questions([SAMPLE])
        with self.assertRaises(ValueError):
            plan_requests(qs, n=6)
        with self.assertRaises(ValueError):
            plan_requests(qs, models=("gpt-x",))

    def test_reasoning_effort_only_for_thinking(self) -> None:
        qs = load_questions([SAMPLE])
        plan = plan_requests(qs, models=("sarashina3.2-mini-thinking",), reasoning_effort="high")
        self.assertEqual({tuple(sorted(r.params.items())) for r in plan}, {(("n", 3), ("reasoning_effort", "high"))})
        # 指定しなければ本文に入れない（API の既定に任せる）。
        plan = plan_requests(qs, models=("sarashina3.2-mini-thinking",))
        self.assertEqual({tuple(r.params) for r in plan}, {("n",)})
        with self.assertRaises(ValueError):
            plan_requests(qs, models=("sarashina3.2-mini-thinking", "sarashina3-mini"), reasoning_effort="low")
        with self.assertRaises(ValueError):
            plan_requests(qs, models=("sarashina3.2-mini-thinking",), reasoning_effort="max")

    def test_estimate_from_rpm(self) -> None:
        qs = load_questions([SAMPLE]) * 1
        many = [q for q in qs for _ in range(5)]  # 15 問
        est = estimate_seconds(plan_requests(many), {"sarashina3-mini": 6, "sarashina3-nano": 60})
        # mini: 15 件 / 6 RPM -> 3 窓 -> 2 回待つ。nano: 15 件 / 60 RPM -> 待たない。
        self.assertEqual(est, {"sarashina3-mini": 122, "sarashina3-nano": 0})


class RunAgainstFakeServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _Handler.received.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp.name)
        self.client = SarashinaClient(
            FAKE_KEY, base_url=self.base_url, clock=FakeClock(START), log_dir=self.log_dir, run_id="run-q"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sample_flows_to_both_models_and_is_logged(self) -> None:
        records = run_plan(self.client, plan_requests(load_questions([SAMPLE])))
        self.assertEqual(len(records), 6)
        self.assertEqual(len(_Handler.received), 6)
        for body in _Handler.received:
            self.assertEqual(body["n"], 3)
            self.assertNotIn("temperature", body)
            self.assertEqual(body["messages"][-1]["role"], "user")
        sent = [(b["model"], r["question_id"]) for b, r in zip(_Handler.received, records)]
        self.assertEqual(
            sorted(sent),
            sorted((m, q) for q in ("math-001", "math-002", "math-003")
                   for m in ("sarashina3-mini", "sarashina3-nano")),
        )
        lines = (self.log_dir / "2027-01-15.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 6)
        logged = json.loads(lines[0])
        self.assertEqual(logged["question_id"], "math-001")
        self.assertEqual(len(logged["response"]["body_json"]["choices"]), 3)

    def test_skip_hook_prevents_send(self) -> None:
        plan = plan_requests(load_questions([SAMPLE]))
        records = run_plan(self.client, plan, should_skip=lambda m, q: m == "sarashina3-mini")
        self.assertEqual(len(records), 3)
        self.assertTrue(all(b["model"] == "sarashina3-nano" for b in _Handler.received))


class CliTest(unittest.TestCase):
    def test_dry_run_lists_and_estimates_without_sending(self) -> None:
        out = io.StringIO()
        with mock.patch("sarashina_eval.client.SarashinaClient") as client_cls, redirect_stdout(out):
            code = run_cli.main([str(SAMPLE), "--dry-run"])
        self.assertEqual(code, 0)
        client_cls.assert_not_called()
        text = out.getvalue()
        self.assertIn("送るリクエスト 6 件", text)
        self.assertIn("math-003", text)
        self.assertIn("sarashina3-mini    3 件 / 30 RPM -> 約 0 秒", text)
        self.assertEqual(text, format_dry_run(plan_requests(load_questions([SAMPLE]))) + "\n")

    def test_dry_run_with_reasoning_effort(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = run_cli.main([str(SAMPLE), "--dry-run", "--models", "sarashina3.2-mini-thinking",
                                 "--reasoning-effort", "medium"])
        self.assertEqual(code, 0)
        self.assertIn("reasoning_effort=medium", out.getvalue())
        err = io.StringIO()
        with redirect_stderr(err):
            code = run_cli.main([str(SAMPLE), "--dry-run", "--reasoning-effort", "medium"])
        self.assertEqual(code, 2)
        self.assertIn("reasoning_effort", err.getvalue())

    def test_resume_reuses_run_id_and_skips_completed(self) -> None:
        out = io.StringIO()
        log_dir = Path(tempfile.mkdtemp())
        with mock.patch("sarashina_eval.client.SarashinaClient") as client_cls, \
                mock.patch.object(run_cli, "run_plan", return_value=[]) as run_plan_mock, \
                redirect_stdout(out):
            client_cls.return_value.run_id = "run-X"
            argv = [str(SAMPLE), "--resume", "run-X", "--log-dir", str(log_dir), "--note", "再開の確認"]
            code = run_cli.main(argv)
        self.assertEqual(code, 0)
        runs = [json.loads(l) for l in (log_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(runs), 1)
        self.assertEqual({k: runs[0][k] for k in ("event", "run_id", "entry", "note", "resumed", "counts", "argv")},
                         {"event": "send", "run_id": "run-X", "entry": "sarashina_eval.run", "note": "再開の確認",
                          "resumed": True, "counts": {"sent": 0}, "argv": argv})
        self.assertEqual(client_cls.call_args.kwargs["run_id"], "run-X")
        # 送った問題集の写しが run_id ごとに残る。
        manifest = json.loads((log_dir / "runs" / "run-X" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([f["kind"] for f in manifest["files"]], ["questions"])
        snap = (log_dir / "runs" / "run-X" / manifest["files"][0]["path"]).read_text(encoding="utf-8")
        self.assertEqual(snap, SAMPLE.read_text(encoding="utf-8"))
        skip = run_plan_mock.call_args.kwargs["should_skip"]
        skip("sarashina3-mini", "math-001")
        client_cls.return_value.is_completed.assert_called_once_with("sarashina3-mini", "math-001")

    def test_broken_file_exits_2_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_text("not json\n", encoding="utf-8")
            err = io.StringIO()
            with mock.patch("sarashina_eval.client.SarashinaClient") as client_cls, redirect_stderr(err):
                code = run_cli.main([str(p), "--dry-run"])
        self.assertEqual(code, 2)
        client_cls.assert_not_called()
        self.assertIn(f"{p}:1:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
