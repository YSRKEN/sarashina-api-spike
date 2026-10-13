"""レートを守って Sarashina API に送り、1 回の呼び出しを JSONL 1 行に記録する。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sarashina_eval.clock import Clock, SystemClock
from sarashina_eval.models import BASE_URL, MODEL_RPM, Endpoint
from sarashina_eval.rate_limiter import RateLimiter

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SEC = 60


class ApiKeyMissingError(RuntimeError):
    """API キーが空のときに送らずに止めるための例外。"""


def load_api_key(env_path: Path | None = None) -> str:
    """API キーを環境変数 ``SARASHINA_API_KEY``、なければ ``.env`` から読む。

    Parameters
    ----------
    env_path : Path, optional
        ``.env`` のパス。省略時はリポジトリ直下の ``.env``。

    Returns
    -------
    str
        API キー。見つからなければ空文字列。
    """
    key = os.environ.get("SARASHINA_API_KEY", "").strip()
    if key:
        return key
    path = env_path if env_path is not None else ROOT / ".env"
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == "SARASHINA_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""


def _iso_utc(ts: float) -> str:
    """UNIX 秒を UTC・ミリ秒の ISO 8601 にする。"""
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """記録用に ``Authorization`` を ``Bearer ***`` に伏せた写しを返す。"""
    return {k: ("Bearer ***" if k.lower() == "authorization" else v) for k, v in headers.items()}


class SarashinaClient:
    """レートを守って Sarashina API に送り、呼び出しごとに記録する実行クラス。

    Parameters
    ----------
    api_key : str, optional
        API キー。省略時は ``load_api_key()`` で読む。空なら ``ApiKeyMissingError``。
    base_url : str, optional
        エンドポイントの基底。既定は ``BASE_URL``。テストでは偽のサーバーへ向ける。
    clock : Clock, optional
        現在時刻と待機の取り出し口。既定は ``SystemClock``。
    log_dir : Path, optional
        記録の置き場。既定はリポジトリ直下の ``logs/``。
    run_id : str, optional
        この実行の識別子。省略時は UUID を振る。
    rpm_by_model : dict[str, int], optional
        モデルごとの上限回数。既定は ``MODEL_RPM``。
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        clock: Clock | None = None,
        log_dir: Path | None = None,
        run_id: str | None = None,
        rpm_by_model: dict[str, int] | None = None,
    ) -> None:
        key = load_api_key() if api_key is None else api_key.strip()
        if not key:
            raise ApiKeyMissingError(
                "SARASHINA_API_KEY が空なので送らない（環境変数か .env に設定すること）"
            )
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._clock = clock if clock is not None else SystemClock()
        self._log_dir = log_dir if log_dir is not None else ROOT / "logs"
        self.run_id = run_id if run_id is not None else str(uuid.uuid4())
        self._limiter = RateLimiter(rpm_by_model or MODEL_RPM, self._clock)

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        question_id: str | None = None,
        attempt: int = 1,
        **params: Any,
    ) -> dict[str, Any]:
        """Chat（``POST /chat/completions``）に 1 回送り、記録を返す。

        Parameters
        ----------
        model : str
            モデル名。
        messages : list of dict
            OpenAI 互換の ``messages``。
        question_id : str, optional
            設問の識別子（記録用）。
        attempt : int, optional
            同じ設問への何回目の送信か（記録用）。
        **params
            ``max_completion_tokens`` などの追加の本文の項目。

        Returns
        -------
        dict
            JSONL に書いたものと同じ記録。
        """
        body = {"model": model, "messages": messages, **params}
        return self._send(Endpoint.CHAT, model, body, question_id=question_id, attempt=attempt)

    def _send(
        self,
        endpoint: Endpoint,
        model: str,
        body: dict[str, Any],
        *,
        question_id: str | None,
        attempt: int,
    ) -> dict[str, Any]:
        """窓を守って POST し、往復をすべて記録する（各エンドポイント共通）。"""
        url = self._base_url + endpoint.value
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        waited = self._limiter.acquire(model)

        record: dict[str, Any] = {
            "run_id": self.run_id,
            "question_id": question_id,
            "attempt": attempt,
            "wait_before_send_sec": round(waited, 3),
            "is_retry_after_429": False,
            "request": {"method": "POST", "url": url, "headers": _redact_headers(headers), "body": body},
        }
        req = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST"
        )
        sent = self._clock.now()
        error = None
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                status, resp_headers, raw = resp.status, list(resp.headers.items()), resp.read()
        except urllib.error.HTTPError as e:
            status, resp_headers, raw = e.code, list(e.headers.items()), e.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            status, resp_headers, raw = None, [], b""
            error = {"type": type(e).__name__, "message": str(e)}
        received = self._clock.now()

        text = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text else None
        except json.JSONDecodeError:
            parsed = None
        record.update(
            {
                "sent_at": _iso_utc(sent),
                "received_at": _iso_utc(received),
                "elapsed_ms": round((received - sent) * 1000, 1),
                # 同名の重複を保つため、[名前, 値] の並びのまま残す。
                "response": {
                    "status": status,
                    "headers": [list(h) for h in resp_headers],
                    "body_text": text,
                    "body_json": parsed,
                },
                "error": error,
            }
        )
        self._append_log(record, sent)
        return record

    def _append_log(self, record: dict[str, Any], sent: float) -> Path:
        """送信日（UTC）の ``YYYY-MM-DD.jsonl`` に 1 行で追記する。"""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        path = self._log_dir / f"{datetime.fromtimestamp(sent, timezone.utc):%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
