"""実行ごとの写し（``<log-dir>/runs/<run_id>/``）の保存と、採点での読み戻しを確かめる。"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sarashina_eval import code_runner, non_chat, score
from sarashina_eval.non_chat_cases import load_cases
from sarashina_eval.run_snapshot import save_snapshot, snapshot_dir, snapshot_paths

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUN_ID = "abcdef01-snap"


def _q(qid: str, expected: str) -> dict:
    return {"id": qid, "category": "math", "prompt": "p", "expected": expected,
            "grading": {"method": "numeric"}, "intent": "i"}


class SaveSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.logs = self.dir / "logs"

    def test_questions_only_selected_lines_with_source(self) -> None:
        qdir = self.dir / "questions"
        qdir.mkdir()
        (qdir / "math.jsonl").write_text("\n".join(json.dumps(_q(i, "3")) for i in ("q1", "q2")), encoding="utf-8")
        out = save_snapshot(self.logs, RUN_ID, questions=([qdir], {"q2"}))
        self.assertEqual(out, self.logs / "runs" / RUN_ID)
        lines = (out / "questions" / "math.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(l)["id"] for l in lines], ["q2"])
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["files"], [{"path": "questions/math.jsonl",
                                              "source": str((qdir / "math.jsonl").resolve()), "kind": "questions"}])
        self.assertEqual(snapshot_paths(self.logs, RUN_ID, "questions"), [out / "questions" / "math.jsonl"])
        self.assertIsNone(snapshot_paths(self.logs, RUN_ID, "inputs"))

    def test_inputs_keep_dir_names_and_filter_kinds(self) -> None:
        pub = self.dir / "inputs"
        priv = self.dir / "inputs_private"
        shutil.copytree(FIXTURES / "inputs", pub)
        priv.mkdir()
        row = json.loads((pub / "guard.jsonl").read_text(encoding="utf-8").splitlines()[0])
        row["id"] = "guard-priv-01"
        (priv / "guard.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        cases = load_cases([pub, priv], ["guard"])
        by_kind: dict[str, set[str]] = {}
        for c in cases:
            by_kind.setdefault(c.kind, set()).add(c.id)
        out = save_snapshot(self.logs, RUN_ID, inputs=([pub, priv], by_kind))
        self.assertTrue((out / "inputs" / "guard.jsonl").is_file())
        self.assertTrue((out / "inputs_private" / "guard.jsonl").is_file())
        self.assertFalse((out / "inputs" / "rerank.jsonl").exists())
        snap = snapshot_paths(self.logs, RUN_ID, "inputs")
        self.assertEqual(snap, [out / "inputs", out / "inputs_private"])
        self.assertEqual(sorted(c.id for c in load_cases(snap)), sorted(c.id for c in cases))

    def test_existing_snapshot_is_not_overwritten_and_warns(self) -> None:
        qfile = self.dir / "q.jsonl"
        qfile.write_text(json.dumps(_q("q1", "3")), encoding="utf-8")
        out = save_snapshot(self.logs, RUN_ID, questions=([qfile], {"q1"}))
        before = (out / self.dir.name / "q.jsonl").read_text(encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            save_snapshot(self.logs, RUN_ID, questions=([qfile], {"q1"}))
        self.assertEqual(err.getvalue(), "")
        qfile.write_text(json.dumps(_q("q1", "4")), encoding="utf-8")
        with contextlib.redirect_stderr(err):
            save_snapshot(self.logs, RUN_ID, questions=([qfile], {"q1"}))
        self.assertIn("警告", err.getvalue())
        self.assertEqual((out / self.dir.name / "q.jsonl").read_text(encoding="utf-8"), before)


class ScoreUsesSnapshotTest(unittest.TestCase):
    """写しがあれば、今の問題集が変わっていても写しと照合する。"""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.logs = self.dir / "logs"
        self.logs.mkdir()
        rec = {"run_id": RUN_ID, "question_id": "q1", "attempt": 1, "sent_at": "2026-10-11T00:00:00+00:00",
               "request": {"body": {"model": "m"}},
               "response": {"status": 200, "body_json": {"choices": [{"message": {"content": "3"}}]}}}
        (self.logs / "2026-10-11.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

    def _score(self, *extra: str) -> tuple[int, str]:
        err = io.StringIO()
        argv = ["--run-id", RUN_ID, "--log-dir", str(self.logs), "--reports-dir", str(self.dir / "reports"), *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err), \
                mock.patch.object(code_runner, "docker_available", return_value=False):
            code = score.main(argv)
        return code, err.getvalue()

    def _last_run(self) -> dict:
        return json.loads((self.logs / "runs.jsonl").read_text(encoding="utf-8").splitlines()[-1])

    def test_snapshot_is_default(self) -> None:
        qfile = self.dir / "q.jsonl"
        qfile.write_text(json.dumps(_q("q1", "3")), encoding="utf-8")
        save_snapshot(self.logs, RUN_ID, questions=([qfile], {"q1"}))
        qfile.write_text(json.dumps(_q("q1", "999")), encoding="utf-8")  # あとで直した
        code, err = self._score()
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        row = self._last_run()
        self.assertEqual(row["counts"].get("correct"), 1)
        self.assertEqual(row["reference"]["source"], "snapshot")
        self.assertTrue(row["reference"]["paths"][0].startswith(str(snapshot_dir(self.logs, RUN_ID))))
        # 明示すれば今の問題集と照合する。
        self._score("--questions", str(qfile))
        self.assertEqual(self._last_run()["reference"], {"source": "current", "paths": [str(qfile)]})
        self.assertNotIn("correct", self._last_run()["counts"])

    def test_no_snapshot_falls_back_with_notice(self) -> None:
        (self.dir / "questions").mkdir()
        (self.dir / "questions" / "q.jsonl").write_text(json.dumps(_q("q1", "3")), encoding="utf-8")
        with mock.patch.object(score, "ROOT", self.dir):
            code, err = self._score()
        self.assertEqual(code, 0)
        self.assertIn("写しが無い", err)
        self.assertEqual(self._last_run()["reference"]["source"], "current")


class NonChatScoreSnapshotTest(unittest.TestCase):
    def test_score_without_inputs_uses_snapshot_or_falls_back(self) -> None:
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        logs = d / "logs"
        cases = load_cases([FIXTURES / "inputs"], ["rerank"])
        save_snapshot(logs, RUN_ID, inputs=([FIXTURES / "inputs"], {"rerank": {c.id for c in cases}}))
        captured: dict = {}

        def fake_score(log_dir, run_id, cs):
            captured["ids"] = [c.id for c in cs]
            return []

        err = io.StringIO()
        with mock.patch.object(non_chat, "score_cases", fake_score), contextlib.redirect_stderr(err):
            non_chat.main(["score", "--run-id", RUN_ID, "--log-dir", str(logs)])
        self.assertEqual(captured["ids"], [c.id for c in cases])
        self.assertNotIn("写しが無い", err.getvalue())
        with mock.patch.object(non_chat, "score_cases", fake_score), \
                mock.patch.object(non_chat, "DEFAULT_INPUT_DIRS", (FIXTURES / "inputs",)), \
                contextlib.redirect_stderr(err):
            non_chat.main(["score", "--run-id", "other", "--log-dir", str(logs)])
        self.assertIn("写しが無い", err.getvalue())


if __name__ == "__main__":
    unittest.main()
