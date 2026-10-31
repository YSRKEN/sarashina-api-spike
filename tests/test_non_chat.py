"""Chat 以外の 4 モデル: 偽のサーバーで要求の形・応答の解釈・採点・入口を確かめる。"""

from __future__ import annotations

import contextlib
import csv
import functools
import io
import json
import tempfile
import threading
import unittest
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from sarashina_eval import client as client_mod
from sarashina_eval import non_chat
from sarashina_eval.client import SarashinaClient
from sarashina_eval.non_chat_cases import CaseSetError, estimate_seconds, load_cases, plan_calls
from sarashina_eval.non_chat_score import (
    CORRECT,
    MISSING,
    PARSE_FAILED,
    UNGRADED,
    WRONG,
    cosine,
    grade_embedding,
    grade_guard,
    grade_rerank,
    parse_guard,
    score_cases,
    write_outputs,
)
from sarashina_eval.run_log import completed_attempts, recent_sends

FAKE_KEY = "test-key-not-real"
START = 1_800_000_000.0
REPO_INPUTS = Path(__file__).resolve().parent.parent / "inputs"
# 件数や中身を決め打ちするテストは、本物の入力ではなくテスト専用の見本を読む。
FIXTURE_INPUTS = Path(__file__).resolve().parent / "fixtures" / "inputs"

#: 偽のベクトル: 文に「暑」「気温」「富士山」が入れば近い向き、それ以外は直交する向き。
def _fake_vector(text: str) -> list[float]:
    if any(k in text for k in ("暑", "気温", "富士山")):
        return [1.0, 0.1, 0.0]
    return [0.0, 0.0, 1.0]


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
    """パスごとに公式の応答例の形で返す。"""

    received: list[dict] = []
    guard_reply = "安全"

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.received.append({"path": self.path, "body": body})
        if self.path.endswith("/chat/completions"):
            out = {"choices": [{"message": {"role": "assistant", "content": _Handler.guard_reply}}]}
        elif self.path.endswith("/embeddings"):
            out = {"object": "list", "model": body["model"],
                   "data": [{"object": "embedding", "index": 0, "embedding": _fake_vector(body["input"])}]}
        elif self.path.endswith("/rerank"):
            # 文書の位置の逆順を「スコアの降順」として返す（わざと順不同で並べる）。
            n = len(body["documents"])
            result = [{"document_id": i, "score": (i + 1) / (n + 1)} for i in range(n)]
            out = {"object": "list", "data": [{"object": "reranked_document_list", "index": 0, "result": result}]}
        else:
            out = {"translated_text": "TRANSLATED:" + body["text"]}
        payload = json.dumps(out, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


class _ServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _Handler.received.clear()
        _Handler.guard_reply = "安全"
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log_dir = self.dir / "logs"
        self.clock = FakeClock(START)
        self.client = SarashinaClient(FAKE_KEY, base_url=self.base_url, clock=self.clock,
                                      log_dir=self.log_dir, run_id="run-nc")

    def tearDown(self) -> None:
        self.tmp.cleanup()


class ClientEndpointTest(_ServerCase):
    """達成条件: client に embeddings・rerank・translate を足し、共通の _send で送る。"""

    def test_embeddings_request_shape(self) -> None:
        rec = self.client.embeddings("今日は暑い", "SEMANTIC_SIMILARITY", question_id="e:anchor")
        got = _Handler.received[0]
        self.assertEqual(got["path"], "/v1/embeddings")
        self.assertEqual(got["body"], {"model": "sarashina3-embedding", "input": "今日は暑い",
                                       "task_type": "SEMANTIC_SIMILARITY"})
        self.assertEqual(rec["response"]["body_json"]["data"][0]["embedding"], [1.0, 0.1, 0.0])
        self.assertEqual(rec["question_id"], "e:anchor")

    def test_embeddings_rejects_bad_input_without_sending(self) -> None:
        with self.assertRaises(ValueError):
            self.client.embeddings(["a", "b"], "SEMANTIC_SIMILARITY")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.client.embeddings("a", "UNKNOWN")
        self.assertEqual(_Handler.received, [])

    def test_rerank_request_shape(self) -> None:
        self.client.rerank("q", ["a", "b"], task_type="GENERAL_QA", question_id="r1")
        self.client.rerank("q", ["a"])
        self.assertEqual(_Handler.received[0]["path"], "/v1/rerank")
        self.assertEqual(_Handler.received[0]["body"], {"model": "sarashina3-rerank", "query": "q",
                                                        "documents": ["a", "b"], "task_type": "GENERAL_QA"})
        self.assertNotIn("task_type", _Handler.received[1]["body"])

    def test_rerank_rejects_bad_input_without_sending(self) -> None:
        with self.assertRaises(ValueError):
            self.client.rerank("q", [])
        with self.assertRaises(ValueError):
            self.client.rerank("q", ["d"] * 21)
        with self.assertRaises(ValueError):
            self.client.rerank("q", ["d"], task_type="CUSTOMIZE")
        self.assertEqual(_Handler.received, [])

    def test_translate_body_has_model_for_window_and_resume(self) -> None:
        self.client.translate("こんにちは", "ja", "en", question_id="t1")
        got = _Handler.received[0]
        self.assertEqual(got["path"], "/v1/translate")
        self.assertEqual(got["body"], {"model": "sarashina3-translate", "text": "こんにちは",
                                       "source_lang": "ja", "target_lang": "en"})
        # 記録から窓の数え直しと済みの判定ができる。
        self.assertIn("sarashina3-translate", recent_sends(self.log_dir, self.clock.now()))
        self.assertIn(("sarashina3-translate", "t1", 1), completed_attempts(self.log_dir, "run-nc"))

    def test_translate_rejects_bad_langs_and_length(self) -> None:
        for s, t in (("ja", "ja"), ("ja", "fr")):
            with self.assertRaises(ValueError):
                self.client.translate("x", s, t)
        with self.assertRaises(ValueError):
            self.client.translate("あ" * 5001, "ja", "en")
        self.assertEqual(_Handler.received, [])

    def test_translate_window_is_six_per_minute(self) -> None:
        for i in range(7):
            self.client.translate("x", "en", "ja", question_id=f"t{i}")
        self.assertEqual(self.clock.sleeps, [61.0])


class GradeTest(unittest.TestCase):
    """達成条件: 採点（guard・embedding・rerank）。"""

    def test_parse_guard(self) -> None:
        self.assertEqual(parse_guard("安全"), ("安全", frozenset()))
        self.assertEqual(parse_guard("有害\nT8, T9"), ("有害", frozenset({"T8", "T9"})))
        self.assertEqual(parse_guard(" 有害\nT12,T1 \n"), ("有害", frozenset({"T12", "T1"})))
        self.assertIsNone(parse_guard("わからない"))
        self.assertIsNone(parse_guard(None))

    def test_grade_guard(self) -> None:
        harm = {"verdict": "有害", "categories": ["T8", "T9"]}
        self.assertEqual(grade_guard({"verdict": "安全"}, "安全")[0], CORRECT)
        self.assertEqual(grade_guard(harm, "有害\nT9, T8")[0], CORRECT)
        status, m = grade_guard(harm, "有害\nT8")
        self.assertEqual((status, m["verdict_ok"]), (WRONG, True))
        self.assertEqual(grade_guard(harm, "安全")[0], WRONG)
        self.assertEqual(grade_guard({"verdict": "安全"}, "有害\nT1")[0], WRONG)
        self.assertEqual(grade_guard(harm, "?")[0], PARSE_FAILED)

    def test_cosine(self) -> None:
        self.assertAlmostEqual(cosine([1, 0], [2, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 3]), 0.0)
        with self.assertRaises(ValueError):
            cosine([1, 0], [1, 0, 0])

    def test_grade_embedding(self) -> None:
        v = {"anchor": [1, 0], "near0": [0.9, 0.1], "far0": [0, 1], "far1": [0.2, 1]}
        status, m = grade_embedding(v, 1, 2)
        self.assertEqual(status, CORRECT)
        self.assertGreater(m["margin"], 0)
        v["near0"] = [0, 1]
        self.assertEqual(grade_embedding(v, 1, 2)[0], WRONG)
        self.assertEqual(grade_embedding({"anchor": [1, 0]}, 1, 1)[0], PARSE_FAILED)

    def test_grade_rerank(self) -> None:
        body = {"data": [{"result": [{"document_id": 2, "score": 0.9}, {"document_id": 0, "score": 0.5},
                                     {"document_id": 1, "score": 0.7}]}]}
        status, m = grade_rerank({"expected_top": 2, "expected_order": [2, 1, 0]}, body)
        self.assertEqual((status, m["order_match"], m["order"]), (CORRECT, True, "2 1 0"))
        status, m = grade_rerank({"expected_top": 0}, body)
        self.assertEqual(status, WRONG)
        self.assertNotIn("order_match", m)
        self.assertEqual(grade_rerank({"expected_top": 0}, {"data": []})[0], PARSE_FAILED)


class CasesTest(unittest.TestCase):
    """達成条件: 入力の形・公開しない置き場・dry-run の見積もり。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, sub: str, kind: str, rows: list[dict]) -> Path:
        d = self.dir / sub
        d.mkdir(exist_ok=True)
        (d / f"{kind}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                         encoding="utf-8")
        return d

    def test_repo_inputs_are_valid(self) -> None:
        cases = load_cases([REPO_INPUTS])
        self.assertEqual({c.kind for c in cases}, {"guard", "embedding", "rerank", "translate"})

    def test_reads_public_and_private_dirs(self) -> None:
        pub = self._write("inputs", "guard", [{"id": "g1", "aim": "a", "input": "x", "expected": {"verdict": "安全"}}])
        priv = self._write("inputs_private", "guard",
                           [{"id": "g2", "aim": "a", "input": "y", "expected": {"verdict": "有害", "categories": ["T1"]}}])
        cases = load_cases([pub, priv, self.dir / "none"])
        self.assertEqual([(c.id, c.source) for c in cases], [("g1", "inputs"), ("g2", "inputs_private")])

    def test_duplicate_id_across_dirs_is_error(self) -> None:
        row = {"id": "g1", "input": "x", "expected": {"verdict": "安全"}}
        pub, priv = self._write("inputs", "guard", [row]), self._write("inputs_private", "guard", [row])
        with self.assertRaises(CaseSetError):
            load_cases([pub, priv])

    def test_invalid_rows_are_errors(self) -> None:
        bad = [
            ("guard", {"id": "g", "input": "x", "expected": {"verdict": "有害"}}),
            ("guard", {"id": "g", "input": "x", "expected": {"verdict": "有害", "categories": ["T13"]}}),
            ("embedding", {"id": "e", "anchor": "a", "near": ["b"], "far": ["c"]}),
            ("rerank", {"id": "r", "query": "q", "documents": ["a"], "expected_top": 1}),
            ("rerank", {"id": "r", "query": "q", "documents": ["a", "b"], "expected_top": 0, "expected_order": [1, 0]}),
            ("translate", {"id": "t", "text": "x", "source_lang": "en", "target_lang": "en"}),
            ("translate", {"id": "a:b", "text": "x", "source_lang": "en", "target_lang": "ja"}),
        ]
        for kind, row in bad:
            with self.subTest(row=row), self.assertRaises(CaseSetError):
                load_cases([self._write("inputs", kind, [row])], [kind])

    def test_plan_and_estimate(self) -> None:
        plan = plan_calls(load_cases([FIXTURE_INPUTS]))
        ids = [c.question_id for c in plan if c.kind == "embedding"]
        self.assertEqual(ids, ["emb-tmp-01:anchor", "emb-tmp-01:near0", "emb-tmp-01:far0",
                               "emb-tmp-02:anchor", "emb-tmp-02:near0", "emb-tmp-02:far0", "emb-tmp-02:far1"])
        emb2 = [c for c in plan if c.question_id.startswith("emb-tmp-02")]
        self.assertEqual([c.args[1] for c in emb2],
                         ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "RETRIEVAL_DOCUMENT", "RETRIEVAL_DOCUMENT"])
        # guard 7 件なら 6 RPM で 1 回窓を待つ。
        from sarashina_eval.non_chat_cases import PlannedCall
        seven = [PlannedCall("guard", "sarashina3-guard", f"g{i}", "chat") for i in range(7)]
        self.assertEqual(estimate_seconds(seven), {"sarashina3-guard": 61})

    def test_dry_run_does_not_send(self) -> None:
        out = io.StringIO()
        with mock.patch.object(client_mod, "SarashinaClient") as cls, contextlib.redirect_stdout(out):
            log_dir = Path(tempfile.mkdtemp())
            code = non_chat.main(["run", "--dry-run", "--inputs", str(FIXTURE_INPUTS), "--log-dir", str(log_dir)])
        self.assertEqual(code, 0)
        cls.assert_not_called()
        self.assertFalse((log_dir / "runs.jsonl").exists())
        self.assertIn("送る要求 13 件", out.getvalue())
        self.assertIn("sarashina3-embedding", out.getvalue())


class EndToEndTest(_ServerCase):
    """達成条件: 入口で流して採点し、logs/ に本文、results/ に本文なしの CSV を出す。"""

    def _main(self, *argv: str) -> tuple[int, str]:
        factory = functools.partial(SarashinaClient, FAKE_KEY, base_url=self.base_url, clock=self.clock)
        out = io.StringIO()
        with mock.patch.object(client_mod, "SarashinaClient", factory), contextlib.redirect_stdout(out):
            code = non_chat.main([*argv, "--inputs", str(FIXTURE_INPUTS), "--log-dir", str(self.log_dir)])
        return code, out.getvalue()

    def test_run_score_and_resume(self) -> None:
        code, out = self._main("run", "--resume", "run-e2e")
        self.assertEqual(code, 0)
        self.assertIn("送信 13 件", out)
        # 再開すると済んだ要求は送らない。
        code, out = self._main("run", "--resume", "run-e2e")
        self.assertIn("送信 0 件", out)
        self.assertEqual(len(_Handler.received), 13)
        runs = [json.loads(l) for l in (self.log_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["event"], r["counts"], r["resumed"]) for r in runs],
                         [("send", {"sent": 13}, True), ("send", {"sent": 0}, True)])
        self.assertEqual(runs[0]["entry"], "sarashina_eval.non_chat run")
        self.assertIsNone(runs[0]["note"])
        # 出力は API の記録の日付ファイルと、末尾に入力の写しの置き場。
        self.assertTrue(all(p.endswith(".jsonl") for p in runs[0]["outputs"][:-1]))
        self.assertEqual(runs[0]["outputs"][-1], str(self.log_dir / "runs" / "run-e2e"))
        # 実行の一覧は API の記録として読まれない（再開の判定を崩さない）。
        self.assertEqual(len(completed_attempts(self.log_dir, "run-e2e")), 13)

        results = self.dir / "results"
        code, out = self._main("score", "--run-id", "run-e2e", "--results-dir", str(results))
        self.assertEqual(code, 0)
        scores = [json.loads(l) for l in (self.log_dir / "scores" / "run-e2e-non-chat.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        by_id = {s["case_id"]: s for s in scores}
        self.assertEqual(by_id["guard-tmp-01"]["status"], CORRECT)
        self.assertEqual(by_id["emb-tmp-01"]["status"], CORRECT)
        self.assertEqual(by_id["emb-tmp-02"]["status"], CORRECT)
        # 偽のサーバーは位置の大きい文書ほど高いスコアを返す。
        self.assertEqual(by_id["rerank-tmp-01"]["metrics"]["top"], 2)
        self.assertEqual(by_id["rerank-tmp-01"]["status"], WRONG)
        self.assertEqual(by_id["tr-tmp-01"]["status"], UNGRADED)
        self.assertEqual(by_id["tr-tmp-01"]["content"], "TRANSLATED:明日は雨が降るでしょう。")
        csv_files = list(results.glob("*-non-chat-scores.csv"))
        self.assertEqual(len(csv_files), 1)
        runs = [json.loads(l) for l in (self.log_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["event"] for r in runs][-1], "score")
        self.assertEqual(runs[-1]["entry"], "sarashina_eval.non_chat score")
        self.assertEqual(runs[-1]["counts"]["cases"], 8)
        self.assertIn(str(csv_files[0]), runs[-1]["outputs"])
        self.assertTrue(csv_files[0].name.endswith("-run-e2e-non-chat-scores.csv"))
        text = csv_files[0].read_text(encoding="utf-8")
        self.assertNotIn("TRANSLATED", text)
        self.assertNotIn("明日は雨", text)
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(rows), 8)
        self.assertEqual(list(rows[0])[:3], ["run_id", "run_date", "kind"])
        self.assertEqual(rows[0]["run_id"], "run-e2e")

    def test_missing_records(self) -> None:
        cases = load_cases([FIXTURE_INPUTS], ["rerank"])
        scores = score_cases(self.log_dir, "nothing", cases)
        self.assertEqual({s.status for s in scores}, {MISSING})

    def test_write_outputs_paths(self) -> None:
        jsonl, csv_path = write_outputs([], "r", self.dir / "s", self.dir / "rep", date(2026, 10, 13))
        self.assertEqual(jsonl.name, "r-non-chat.jsonl")
        self.assertEqual(csv_path.name, "2026-10-13-r-non-chat-scores.csv")


if __name__ == "__main__":
    unittest.main()
