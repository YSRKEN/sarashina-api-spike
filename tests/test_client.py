"""偽のサーバーへ送り、レートの待ちと記録の中身を確かめる。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from sarashina_eval import MODEL_RPM, ApiKeyMissingError, SarashinaClient, load_api_key

FAKE_KEY = "test-key-not-real"
START = 1_800_000_000.0  # 2027-01-15 08:00:00 UTC


#: 窓の待ちを確かめるための上限（公式の値とは別に固定する）。
WINDOW_RPM = {"sarashina3-mini": 6, "sarashina3-nano": 60}


class FakeClock:
    """sleep で時刻を進めるだけの偽の時計。実際には待たない。"""

    def __init__(self, start: float) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class _Handler(BaseHTTPRequestHandler):
    """受けた本文を記録し、重複ヘッダー付きの JSON を返す。"""

    received: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        _Handler.received.append(
            {"path": self.path, "auth": self.headers["Authorization"], "body": json.loads(self.rfile.read(length))}
        )
        payload = json.dumps({"choices": [{"message": {"content": "はい"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Dup", "a")
        self.send_header("X-Dup", "b")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


class ClientTest(unittest.TestCase):
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
        self.clock = FakeClock(START)
        self.client = SarashinaClient(
            FAKE_KEY, base_url=self.base_url, clock=self.clock, log_dir=self.log_dir, run_id="run-1",
            rpm_by_model=WINDOW_RPM,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _chat(self, model: str = "sarashina3-mini", **kw):
        return self.client.chat(model, [{"role": "user", "content": "こんにちは"}], **kw)

    def test_rpm_settings(self) -> None:
        self.assertEqual(
            MODEL_RPM,
            {
                "sarashina3.2-mini-thinking": 30,
                "sarashina3-mini": 30,
                "sarashina3-guard": 6,
                "sarashina3-translate": 30,
                "sarashina3-nano": 60,
                "sarashina3-embedding": 60,
                "sarashina3-rerank": 60,
            },
        )

    def test_timeout_default_and_override(self) -> None:
        seen = []

        def fake_urlopen(req, timeout):
            seen.append(timeout)
            raise TimeoutError("timed out")

        with mock.patch("sarashina_eval.client.urllib.request.urlopen", fake_urlopen):
            self._chat()
            slow = SarashinaClient(FAKE_KEY, base_url=self.base_url, clock=self.clock, log_dir=self.log_dir,
                                   run_id="run-2", timeout_sec=180)
            rec = slow.chat("sarashina3.2-mini-thinking", [{"role": "user", "content": "x"}])
        self.assertEqual(seen, [60, 180])
        self.assertEqual(rec["error"]["type"], "TimeoutError")
        self.assertEqual(rec["timeout_sec"], 180)

    def test_timeout_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            SarashinaClient(FAKE_KEY, base_url=self.base_url, clock=self.clock, log_dir=self.log_dir, timeout_sec=0)

    def test_mini_seventh_send_waits_until_61s_after_first(self) -> None:
        sent_times = []
        for i in range(7):
            rec = self._chat(question_id=f"q{i}")
            sent_times.append(self.clock.now())
            if i < 6:
                self.assertEqual(rec["wait_before_send_sec"], 0.0)
        self.assertEqual(self.clock.sleeps, [61.0])
        self.assertGreaterEqual(sent_times[6] - START, 61.0)
        self.assertEqual(rec["wait_before_send_sec"], 61.0)
        self.assertEqual(len(_Handler.received), 7)

    def test_window_is_per_model(self) -> None:
        for _ in range(6):
            self._chat("sarashina3-mini")
        rec = self._chat("sarashina3-nano")
        self.assertEqual(rec["wait_before_send_sec"], 0.0)
        self.assertEqual(self.clock.sleeps, [])

    def test_no_wait_after_window_passes(self) -> None:
        for _ in range(6):
            self._chat()
        self.clock.t += 61
        self._chat()
        self.assertEqual(self.clock.sleeps, [])

    def test_log_record(self) -> None:
        rec = self._chat(question_id="q1", max_completion_tokens=8)
        path = self.log_dir / "2027-01-15.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        logged = json.loads(lines[0])
        self.assertEqual(logged, json.loads(json.dumps(rec, ensure_ascii=False)))
        self.assertEqual(logged["sent_at"], "2027-01-15T08:00:00.000+00:00")
        self.assertEqual(logged["received_at"], "2027-01-15T08:00:00.000+00:00")
        self.assertEqual(logged["elapsed_ms"], 0.0)
        req = logged["request"]
        self.assertEqual(req["method"], "POST")
        self.assertEqual(req["url"], self.base_url + "/chat/completions")
        self.assertEqual(req["headers"]["Authorization"], "Bearer ***")
        self.assertEqual(req["body"]["max_completion_tokens"], 8)
        self.assertNotIn(FAKE_KEY, lines[0])
        resp = logged["response"]
        self.assertEqual(resp["status"], 200)
        self.assertIn(["X-Dup", "a"], resp["headers"])
        self.assertIn(["X-Dup", "b"], resp["headers"])
        self.assertEqual(resp["body_json"]["choices"][0]["message"]["content"], "はい")
        self.assertIsInstance(resp["body_text"], str)
        self.assertIsNone(logged["error"])
        self.assertEqual(logged["run_id"], "run-1")
        self.assertEqual(logged["question_id"], "q1")
        self.assertEqual(logged["attempt"], 1)
        self.assertFalse(logged["is_retry_after_429"])
        # サーバーには本物の（偽の）キーが Bearer で届く。
        self.assertEqual(_Handler.received[0]["auth"], f"Bearer {FAKE_KEY}")

    def test_connection_error_is_logged(self) -> None:
        client = SarashinaClient(
            FAKE_KEY, base_url="http://127.0.0.1:1/v1", clock=self.clock, log_dir=self.log_dir
        )
        rec = client.chat("sarashina3-nano", [{"role": "user", "content": "x"}])
        self.assertIsNone(rec["response"]["status"])
        self.assertEqual(rec["error"]["type"], "URLError")


class ApiKeyTest(unittest.TestCase):
    def test_empty_key_raises(self) -> None:
        with self.assertRaises(ApiKeyMissingError):
            SarashinaClient("  ")

    def test_env_var_wins(self) -> None:
        with mock.patch.dict(os.environ, {"SARASHINA_API_KEY": FAKE_KEY}):
            self.assertEqual(load_api_key(Path("/nonexistent/.env")), FAKE_KEY)

    def test_dotenv_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text('OTHER=1\nSARASHINA_API_KEY="fake-from-dotenv"\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_api_key(env), "fake-from-dotenv")
                self.assertEqual(load_api_key(Path(d) / "missing"), "")


if __name__ == "__main__":
    unittest.main()
