"""再開（窓の数え直し・済みの判定）と送り直しを、偽のサーバーと偽の時計で確かめる。"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from sarashina_eval import RateLimitExceededError, SarashinaClient
from sarashina_eval.client import _iso_utc

FAKE_KEY = "test-key-not-real"
START = 1_800_000_000.0  # 2027-01-15 08:00:00 UTC
MINI = "sarashina3-mini"


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


class _ScriptedHandler(BaseHTTPRequestHandler):
    """``script`` の先頭から順に状態コードを返す。空なら 200。"""

    script: list[int] = []
    hits = 0

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        _ScriptedHandler.hits += 1
        status = _ScriptedHandler.script.pop(0) if _ScriptedHandler.script else 200
        payload = json.dumps({"status": status}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _ScriptedHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _ScriptedHandler.script = []
        _ScriptedHandler.hits = 0
        self.tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp.name)
        self.clock = FakeClock(START)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def client(self, run_id: str = "run-1") -> SarashinaClient:
        # 窓の仕組みを見る試験なので、mini の上限を公式の値（30）ではなく 6 に固定する。
        return SarashinaClient(
            FAKE_KEY, base_url=self.base_url, clock=self.clock, log_dir=self.log_dir, run_id=run_id,
            rpm_by_model={MINI: 6, "sarashina3-nano": 60},
        )

    def chat(self, client: SarashinaClient, **kw):
        return client.chat(MINI, [{"role": "user", "content": "x"}], **kw)

    def logged(self) -> list[dict]:
        out = []
        for p in sorted(self.log_dir.glob("*.jsonl")):
            out += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
        return out


class ResumeWindowTest(_Base):
    """起動時に logs/ の直近 61 秒の送信を窓に数え直す。"""

    def test_restart_counts_recent_sends(self) -> None:
        first = self.client()
        for i in range(6):
            self.chat(first, question_id=f"q{i}")
        self.clock.t += 10
        second = self.client()  # 起動し直し
        rec = self.chat(second, question_id="q6")
        # 最初の送信（START）から 61 秒たつまで待つ。
        self.assertEqual(self.clock.sleeps, [51.0])
        self.assertEqual(rec["wait_before_send_sec"], 51.0)

    def test_old_sends_are_not_counted(self) -> None:
        first = self.client()
        for i in range(6):
            self.chat(first, question_id=f"q{i}")
        self.clock.t += 62
        second = self.client()
        self.chat(second, question_id="q6")
        self.assertEqual(self.clock.sleeps, [])

    def test_window_seed_is_per_model(self) -> None:
        first = self.client()
        for i in range(6):
            self.chat(first, question_id=f"q{i}")
        second = self.client()
        rec = second.chat("sarashina3-nano", [{"role": "user", "content": "x"}])
        self.assertEqual(rec["wait_before_send_sec"], 0.0)


class ResumeCompletedTest(_Base):
    """--resume <run_id> 相当: その run_id で 200 済みの (question_id, attempt) を判定する。"""

    def test_completed_after_restart(self) -> None:
        _ScriptedHandler.script = [200, 400]
        first = self.client("run-A")
        self.chat(first, question_id="q1")
        self.chat(first, question_id="q2")
        resumed = self.client("run-A")
        self.assertTrue(resumed.is_completed(MINI, "q1", 1))
        self.assertFalse(resumed.is_completed(MINI, "q1", 2))
        self.assertFalse(resumed.is_completed(MINI, "q2", 1))  # 400 は済みではない
        self.assertFalse(resumed.is_completed(MINI, "q3", 1))

    def test_other_run_id_is_not_completed(self) -> None:
        self.chat(self.client("run-A"), question_id="q1")
        self.assertFalse(self.client("run-B").is_completed(MINI, "q1", 1))

    def test_completed_is_per_model(self) -> None:
        # 同じ設問を mini と nano の両方に送るので、済みかはモデルごとに分ける。
        self.chat(self.client("run-A"), question_id="q1")
        resumed = self.client("run-A")
        self.assertTrue(resumed.is_completed(MINI, "q1", 1))
        self.assertFalse(resumed.is_completed("sarashina3-nano", "q1", 1))

    def test_completed_within_same_run(self) -> None:
        c = self.client()
        self.assertFalse(c.is_completed(MINI, "q1"))
        self.chat(c, question_id="q1")
        self.assertTrue(c.is_completed(MINI, "q1"))


class Retry429Test(_Base):
    def test_429_waits_61s_and_retries_once(self) -> None:
        _ScriptedHandler.script = [429, 200]
        rec = self.chat(self.client(), question_id="q1")
        self.assertEqual(rec["response"]["status"], 200)
        self.assertEqual(self.clock.sleeps, [61.0])
        logs = self.logged()
        self.assertEqual([r["response"]["status"] for r in logs], [429, 200])
        self.assertEqual([r["is_retry_after_429"] for r in logs], [False, True])
        self.assertEqual(logs[1]["wait_before_send_sec"], 61.0)

    def test_second_429_stops_run(self) -> None:
        _ScriptedHandler.script = [429, 429, 200]
        with self.assertRaises(RateLimitExceededError) as cm:
            self.chat(self.client(), question_id="q1")
        self.assertEqual(cm.exception.record["response"]["status"], 429)
        self.assertEqual(_ScriptedHandler.hits, 2)
        self.assertEqual([r["is_retry_after_429"] for r in self.logged()], [False, True])


class Retry5xxTest(_Base):
    def test_503_retries_with_2_and_4_sec(self) -> None:
        _ScriptedHandler.script = [503, 500, 200]
        rec = self.chat(self.client(), question_id="q1")
        self.assertEqual(rec["response"]["status"], 200)
        self.assertEqual(self.clock.sleeps, [2.0, 4.0])
        self.assertEqual([r["is_retry_after_429"] for r in self.logged()], [False, False, False])

    def test_5xx_gives_up_after_two_retries(self) -> None:
        _ScriptedHandler.script = [503, 503, 503, 200]
        rec = self.chat(self.client(), question_id="q1")
        self.assertEqual(rec["response"]["status"], 503)
        self.assertEqual(_ScriptedHandler.hits, 3)
        self.assertEqual(self.clock.sleeps, [2.0, 4.0])

    def test_client_errors_are_not_retried(self) -> None:
        for status in (400, 401, 404, 413):
            with self.subTest(status=status):
                _ScriptedHandler.hits = 0
                _ScriptedHandler.script = [status]
                rec = self.chat(self.client(), question_id=f"q{status}")
                self.assertEqual(rec["response"]["status"], status)
                self.assertEqual(_ScriptedHandler.hits, 1)
        self.assertEqual(self.clock.sleeps, [])
        self.assertEqual(len(self.logged()), 4)


class RetryCountsInWindowTest(_Base):
    """送り直しも 1 回として窓に数える。"""

    def test_retry_consumes_window(self) -> None:
        c = self.client()
        _ScriptedHandler.script = [503]
        self.chat(c, question_id="q0")  # 送信 2 回（503→200）
        for i in range(1, 5):
            self.chat(c, question_id=f"q{i}")  # 計 6 回
        t_first = START
        self.clock.sleeps.clear()
        rec = self.chat(c, question_id="q5")  # 7 回目は窓で待つ
        self.assertEqual(len(self.clock.sleeps), 1)
        self.assertEqual(_iso_utc(self.clock.now()), _iso_utc(t_first + 61))
        self.assertGreater(rec["wait_before_send_sec"], 0)

    def test_429_retry_goes_through_limiter(self) -> None:
        c = self.client()
        for i in range(5):
            self.chat(c, question_id=f"q{i}")
        _ScriptedHandler.script = [429]
        self.clock.sleeps.clear()
        # 6 回目が 429、61 秒待つと窓は空くので limiter の追加の待ちは無い。
        rec = self.chat(c, question_id="q5")
        self.assertEqual(self.clock.sleeps, [61.0])
        self.assertTrue(rec["is_retry_after_429"])


if __name__ == "__main__":
    unittest.main()
