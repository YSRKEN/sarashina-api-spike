"""Tool Calling の往復・偽の道具・採点を、偽のサーバーで確かめる。"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from sarashina_eval import SarashinaClient, fake_tools, tools

FAKE_KEY = "test-key-not-real"
FIXTURE = Path(__file__).parent / "fixtures" / "tool_scenarios.jsonl"


class FakeClock:
    """待たない時計。"""

    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def now(self) -> float:
        self.t += 0.001
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class _Handler(BaseHTTPRequestHandler):
    """最後が user なら発話に応じて tool_calls を返し、tool なら結果を答えに埋めて返す。

    ``loop`` が真なら、いつも tool_calls を返し続ける（上限の打ち切りを見る）。
    """

    received: list[dict] = []
    loop = False

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.received.append(body)
        last = body["messages"][-1]
        if _Handler.loop or last["role"] == "user" and "天気" in last["content"]:
            msg = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call-{len(body['messages'])}", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": json.dumps({"city": "ほしぞら市", "date": "2026-10-14"}, ensure_ascii=False)}}]}
            finish = "tool_calls"
        elif last["role"] == "tool":
            res = json.loads(last["content"])
            msg = {"role": "assistant", "content": f"{res['weather']}、最高{res['max_c']}度です。", "tool_calls": []}
            finish = "stop"
        else:
            msg = {"role": "assistant", "content": "Thank you", "tool_calls": []}
            finish = "stop"
        payload = json.dumps({"choices": [{"index": 0, "message": msg, "finish_reason": finish}]},
                             ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


class FakeToolsTest(unittest.TestCase):
    def test_known_and_unknown(self) -> None:
        self.assertEqual(json.loads(fake_tools.execute("get_weather", '{"city":"かざみ町","date":"2026-10-15"}'))["weather"], "晴れ")
        self.assertIn("error", json.loads(fake_tools.execute("get_weather", '{"city":"つきみ村","date":"2026-10-14"}')))
        self.assertIn("404", fake_tools.execute("web_fetch", '{"url":"https://example.test/x"}'))
        self.assertIn("error", json.loads(fake_tools.execute("web_fetch", "{壊れた")))
        self.assertIn("error", json.loads(fake_tools.execute("no_such", "{}")))
        self.assertEqual(json.loads(fake_tools.execute("calculator", '{"expression":"(19-9)*2"}'))["result"], 20)
        self.assertIn("error", json.loads(fake_tools.execute("calculator", '{"expression":"__import__(\\"os\\")"}')))


class ToolsFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _Handler.received = []
        _Handler.loop = False
        self.tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp.name)
        self.scenarios = {s.id: s for s in tools.load_scenarios([FIXTURE])}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _client(self, run_id: str = "run-1") -> SarashinaClient:
        return SarashinaClient(FAKE_KEY, base_url=self.base, clock=FakeClock(), log_dir=self.log_dir, run_id=run_id)

    def test_round_trip_and_score(self) -> None:
        recs = tools.run_conversation(self._client(), "sarashina3-nano", self.scenarios["w"], "omit")
        self.assertEqual(len(recs), 2)
        # 1 回目は tool_choice を送らず、2 回目は tool メッセージに tool_call_id を付けて返す。
        self.assertNotIn("tool_choice", _Handler.received[0])
        tool_msg = _Handler.received[1]["messages"][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call-1")
        self.assertEqual([r["attempt"] for r in recs], [1, 2])
        self.assertEqual(recs[0]["question_id"], "w#omit")
        scores, unknown = tools.score_run(self.log_dir, "run-1", self.scenarios)
        self.assertEqual(unknown, [])
        s = scores[0]
        self.assertTrue(s.passed, s)
        self.assertEqual((s.rounds, s.truncated, s.facts_missing), (2, False, []))

    def test_required_first_condition(self) -> None:
        tools.run_conversation(self._client(), "sarashina3-nano", self.scenarios["w"], "required-first")
        self.assertEqual([b.get("tool_choice") for b in _Handler.received], ["required", "auto"])

    def test_no_tool_scenario(self) -> None:
        tools.run_conversation(self._client(), "sarashina3-nano", self.scenarios["none"], "auto")
        s = tools.score_run(self.log_dir, "run-1", self.scenarios)[0][0]
        self.assertTrue(s.passed)
        self.assertEqual(s.calls, [])

    def test_truncated_at_limit(self) -> None:
        _Handler.loop = True
        recs = tools.run_conversation(self._client(), "sarashina3-nano", self.scenarios["none"], "omit", max_rounds=3)
        self.assertEqual(len(recs), 3)
        s = tools.score_run(self.log_dir, "run-1", self.scenarios)[0][0]
        self.assertTrue(s.truncated)
        self.assertFalse(s.passed)
        self.assertEqual(s.unexpected_calls, 3)
        self.assertTrue(tools.is_finished(recs[-1], 3))

    def test_cli_dry_run_resume_and_score(self) -> None:
        self.assertEqual(tools.main(["run", str(FIXTURE), "--dry-run", "--models", "sarashina3-nano"]), 0)
        self.assertEqual(tools.main(["run", str(FIXTURE), "--models", "sarashina3-guard", "--dry-run"]), 2)
        # 送る部分は client を差し替えて回す（API キーを読まない）。
        client = self._client("run-2")
        for m, sc, c in tools.plan(list(self.scenarios.values()), ["sarashina3-nano"], ["omit"]):
            tools.run_conversation(client, m, sc, c)
        n = len(_Handler.received)
        done = {k for k, r in tools.latest_records(self.log_dir, "run-2").items() if tools.is_finished(r, 5)}
        self.assertEqual(len(done), 2)
        self.assertEqual(n, 3)
        reports = self.log_dir / "reports"
        self.assertEqual(tools.main(["score", "--run-id", "run-2", "--log-dir", str(self.log_dir),
                                     "--scenarios", str(FIXTURE), "--reports-dir", str(reports)]), 0)
        csvs = list(reports.glob("*-run-2-tools-scores.csv"))
        self.assertEqual(len(csvs), 1)
        text = csvs[0].read_text(encoding="utf-8")
        self.assertNotIn("最高", text)  # CSV に本文を入れない
        self.assertTrue((self.log_dir / "scores" / "run-2-tools.jsonl").is_file())
        runs = [json.loads(l) for l in (self.log_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(runs[-1]["counts"]["passed"], 2)

    def test_load_rejects_unknown_tool(self) -> None:
        bad = self.log_dir / "bad.jsonl"
        bad.write_text('{"id":"x","prompt":"p","tools":["nope"]}\n', encoding="utf-8")
        with self.assertRaises(tools.ScenarioError):
            tools.load_scenarios([bad])


if __name__ == "__main__":
    unittest.main()
