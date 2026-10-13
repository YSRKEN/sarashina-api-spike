"""``logs/`` の JSONL を読み、再開に要る情報（直近の送信・済んだ問題）を取り出す。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

#: 起動時に窓へ数え直す送信の範囲（秒）。
RESUME_WINDOW_SEC = 61.0

#: 実行の一覧のファイル名（``log_dir`` の直下）。API の記録としては読まない。
RUNS_FILE = "runs.jsonl"


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
        if path.name == RUNS_FILE:
            continue
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


def _log_paths(log_dir: Path) -> list[Path]:
    """API の記録の日付ファイルを名前順で返す（実行の一覧は除く）。"""
    if not log_dir.is_dir():
        return []
    return [p for p in sorted(log_dir.glob("*.jsonl")) if p.name != RUNS_FILE]


def run_log_files(log_dir: Path, run_id: str) -> list[Path]:
    """``run_id`` の記録を含む日付ファイルを集める。

    Parameters
    ----------
    log_dir : Path
        記録の置き場。
    run_id : str
        実行の識別子。

    Returns
    -------
    list of Path
        名前順。記録が無ければ空。
    """
    out = []
    for path in _log_paths(log_dir):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("run_id") == run_id:
                out.append(path)
                break
    return out


def run_date(log_dir: Path, run_id: str) -> date | None:
    """``run_id`` の最初の送信（``sent_at`` の最小）の UTC 日付を返す。

    同じ実行を後日採点し直しても、得点の CSV が同じ名前になるように使う。

    Parameters
    ----------
    log_dir : Path
    run_id : str

    Returns
    -------
    date or None
        送信の記録が無ければ None。
    """
    first: datetime | None = None
    for rec in iter_records(log_dir):
        sent_at = rec.get("sent_at")
        if rec.get("run_id") != run_id or not isinstance(sent_at, str):
            continue
        try:
            ts = datetime.fromisoformat(sent_at)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if first is None or ts < first:
            first = ts
    return first.astimezone(timezone.utc).date() if first else None


def append_run(
    log_dir: Path,
    *,
    event: str,
    run_id: str,
    entry: str,
    note: str | None,
    resumed: bool,
    counts: dict[str, int],
    outputs: list[Path],
    argv: list[str],
    now: datetime | None = None,
) -> dict[str, Any]:
    """実行の一覧 ``<log_dir>/runs.jsonl`` に 1 行足す。

    Parameters
    ----------
    log_dir : Path
        記録の置き場。無ければ作る。
    event : str
        ``send`` か ``score``。
    run_id : str
    entry : str
        どの入口か（例 ``sarashina_eval.run``）。
    note : str or None
        ``--note`` で渡された目的。
    resumed : bool
        ``--resume`` での再開か。
    counts : dict
        件数（送った数、採点の状態ごとの数など）。
    outputs : list of Path
        書いたファイル。
    argv : list of str
        入口に渡された引数。
    now : datetime, optional
        記録する時刻。省略時は今（UTC）。

    Returns
    -------
    dict
        書いた 1 行。

    Raises
    ------
    ValueError
        ``event`` が ``send``・``score`` 以外のとき。
    """
    if event not in ("send", "score"):
        raise ValueError(f"event は send か score: {event!r}")
    at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    row = {
        "event": event,
        "run_id": run_id,
        "at": at.isoformat(timespec="milliseconds"),
        "entry": entry,
        "note": note,
        "resumed": resumed,
        "counts": counts,
        "outputs": [str(p) for p in outputs],
        "argv": list(argv),
    }
    line = json.dumps(row, ensure_ascii=False) + "\n"  # 書けない値なら、ファイルを開く前に落とす
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / RUNS_FILE).open("a", encoding="utf-8") as f:
        f.write(line)
    return row
