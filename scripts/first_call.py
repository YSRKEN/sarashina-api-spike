"""Sarashina API に最小のリクエストを 1 回だけ送り、HTTP の往復をすべて記録する。

レート制限の実測（1 分の区切りの起点、429 の本文の文面）の前提を作るための
使い捨てに近いスクリプトである。送るのは sarashina3-nano への 1 回だけで、
再送はしない。

使い方
------
    python3 scripts/first_call.py            # 送信する
    python3 scripts/first_call.py --dry-run  # 送らずに、送る内容だけ表示する

API キーは環境変数か、リポジトリ直下の ``.env`` の ``SARASHINA_API_KEY`` から読む。
記録は ``logs/YYYY-MM-DD.jsonl`` に 1 行で追記する（``logs/`` は公開しない）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://api.sarashina.ai/v1/chat/completions"
MODEL = "sarashina3-nano"
TIMEOUT_SEC = 60


def load_api_key() -> str:
    """API キーを環境変数、なければ ``.env`` から読む。

    Returns
    -------
    str
        API キー。見つからなければ空文字列。
    """
    key = os.environ.get("SARASHINA_API_KEY", "").strip()
    if key:
        return key
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == "SARASHINA_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""


def utc_now_iso() -> str:
    """現在時刻を UTC・ミリ秒の ISO 8601 で返す。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """記録用に ``Authorization`` だけを伏せたヘッダーの写しを返す。"""
    return {k: ("Bearer ***" if k.lower() == "authorization" else v) for k, v in headers.items()}


def append_log(record: dict) -> Path:
    """記録を日付ごとの JSONL に 1 行で追記し、そのパスを返す。"""
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="送らずに送る内容だけ表示する")
    args = parser.parse_args()

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "こんにちは。一言で返事をしてください。"}],
        "max_completion_tokens": 32,
    }
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    api_key = load_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if args.dry_run:
        print(json.dumps({"url": ENDPOINT, "headers": redact_headers(headers), "body": body},
                         ensure_ascii=False, indent=2))
        return 0
    if not api_key:
        print("SARASHINA_API_KEY が空なので送らずに終了する（.env に書き込むこと）", file=sys.stderr)
        return 2

    record: dict = {
        "run_id": str(uuid.uuid4()),
        "question_id": None,
        "attempt": 1,
        "wait_before_send_sec": 0.0,
        "is_retry_after_429": False,
        "request": {"method": "POST", "url": ENDPOINT, "headers": redact_headers(headers), "body": body},
    }
    req = urllib.request.Request(ENDPOINT, data=body_bytes, headers=headers, method="POST")
    record["sent_at"] = utc_now_iso()
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            status, resp_headers, raw = resp.status, list(resp.headers.items()), resp.read()
        record["error"] = None
    except urllib.error.HTTPError as e:
        status, resp_headers, raw = e.code, list(e.headers.items()), e.read()
        record["error"] = None
    except (urllib.error.URLError, TimeoutError) as e:
        status, resp_headers, raw = None, [], b""
        record["error"] = {"type": type(e).__name__, "message": str(e)}
    record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
    record["received_at"] = utc_now_iso()

    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text) if text else None
    except json.JSONDecodeError:
        parsed = None
    # ヘッダーは同名の重複もありうるので、並びのまま残す。
    record["response"] = {"status": status, "headers": resp_headers, "body_text": text, "body_json": parsed}

    path = append_log(record)
    print(f"status={status} elapsed_ms={record['elapsed_ms']} log={path.relative_to(ROOT)}")
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
