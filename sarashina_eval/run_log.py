"""``logs/`` の JSONL を読み、再開に要る情報（直近の送信・済んだ問題）を取り出す。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

#: 起動時に窓へ数え直す送信の範囲（秒）。
RESUME_WINDOW_SEC = 61.0


def iter_records(log_dir: Path) -> Iterator[dict[str, Any]]:
    """``log_dir`` 下の ``*.jsonl`` の記録を 1 件ずつ返す。

    Parameters
    ----------
    log_dir : Path
        記録の置き場。無ければ何も返さない。

    Yields
    ------
    dict
        1 行分の記録。JSON として読めない行は飛ばす。
    """
    if not log_dir.is_dir():
        return
    for path in sorted(log_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def _model_of(rec: dict[str, Any]) -> str | None:
    """記録の送信本文からモデル名を取り出す。"""
    body = (rec.get("request") or {}).get("body") or {}
    model = body.get("model")
    return model if isinstance(model, str) else None


def recent_sends(
    log_dir: Path, now: float, window_sec: float = RESUME_WINDOW_SEC
) -> dict[str, list[float]]:
    """直近 ``window_sec`` 秒の送信時刻（``sent_at``）をモデルごとに集める。

    Parameters
    ----------
    log_dir : Path
        記録の置き場。
    now : float
        現在時刻（UNIX 秒）。
    window_sec : float, optional
        遡る秒数。既定は 61 秒。

    Returns
    -------
    dict[str, list[float]]
        モデル名ごとの送信時刻（昇順）。
    """
    out: dict[str, list[float]] = {}
    for rec in iter_records(log_dir):
        model, sent_at = _model_of(rec), rec.get("sent_at")
        if model is None or not isinstance(sent_at, str):
            continue
        ts = datetime.fromisoformat(sent_at).timestamp()
        if now - window_sec <= ts <= now:
            out.setdefault(model, []).append(ts)
    for times in out.values():
        times.sort()
    return out


def completed_attempts(log_dir: Path, run_id: str) -> set[tuple[str | None, str | None, int]]:
    """``run_id`` で HTTP 200 が記録済みの ``(model, question_id, attempt)`` を集める。

    同じ設問を複数のモデルに送るので、モデルも鍵に含める。

    Parameters
    ----------
    log_dir : Path
        記録の置き場。
    run_id : str
        再開する実行の識別子。

    Returns
    -------
    set of tuple
        済んだ ``(model, question_id, attempt)`` の集合。
    """
    done: set[tuple[str | None, str | None, int]] = set()
    for rec in iter_records(log_dir):
        if rec.get("run_id") == run_id and (rec.get("response") or {}).get("status") == 200:
            model = ((rec.get("request") or {}).get("body") or {}).get("model")
            done.add((model, rec.get("question_id"), int(rec.get("attempt", 1))))
    return done
