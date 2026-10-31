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
from sarashina_eval.models import (
    BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_TASK_TYPES,
    MODEL_RPM,
    RERANK_MODEL,
    RERANK_TASK_TYPES,
    TRANSLATE_MAX_CHARS,
    TRANSLATE_MODEL,
    Endpoint,
)
from sarashina_eval.rate_limiter import RateLimiter
from sarashina_eval.run_log import completed_attempts, recent_sends

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SEC = 60
#: 429 のあと送り直すまでの秒数（送り直しは 1 回だけ）。
RETRY_AFTER_429_SEC = 61.0
#: 送り直す 5xx。
SERVER_ERROR_STATUSES = frozenset({500, 503})
#: 5xx の送り直しの前に待つ秒数（並びの長さが送り直しの回数）。
SERVER_ERROR_BACKOFF_SEC = (2.0, 4.0)


class ApiKeyMissingError(RuntimeError):
    """API キーが空のときに送らずに止めるための例外。"""


class RateLimitExceededError(RuntimeError):
    """429 のあと送り直してもまた 429 だったときに、実行を止めるための例外。

    Parameters
    ----------
    message : str
        説明。
    record : dict
        2 回目の 429 の記録（JSONL には書き済み）。
    """

    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


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
    timeout_sec : float, optional
        1 リクエストの時間切れの秒数。既定は ``TIMEOUT_SEC``。
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
        timeout_sec: float = TIMEOUT_SEC,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec は正の数（指定は {timeout_sec}）")
        self._timeout_sec = timeout_sec
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
        # 起動し直しても上限を超えないよう、直近の送信を窓に数え直す。
        for model, times in recent_sends(self._log_dir, self._clock.now()).items():
            self._limiter.seed(model, times)
        self._completed = completed_attempts(self._log_dir, self.run_id)

    def is_completed(self, model: str, question_id: str | None, attempt: int = 1) -> bool:
        """この run_id で ``(model, question_id, attempt)`` が HTTP 200 で済んでいるかを返す。

        起動時に記録から読んだものと、この実行で 200 になったものの両方を見る。

        Parameters
        ----------
        model : str
            モデル名。同じ設問でもモデルが違えば別に数える。
        question_id : str or None
            設問の識別子。
        attempt : int, optional
            同じ設問への何回目の送信か。

        Returns
        -------
        bool
            済んでいれば True。
        """
        return (model, question_id, attempt) in self._completed

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

    def embeddings(
        self,
        text: str,
        task_type: str,
        *,
        model: str = EMBEDDING_MODEL,
        question_id: str | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        """Embeddings（``POST /embeddings``）に 1 テキストを送り、記録を返す。

        Parameters
        ----------
        text : str
            ベクトルにするテキスト（1 リクエスト 1 テキスト。配列は送れない）。
        task_type : str
            ``EMBEDDING_TASK_TYPES`` のいずれか（必須）。
        model : str, optional
            既定は ``sarashina3-embedding``。
        question_id : str, optional
            記録用の識別子。
        attempt : int, optional
            同じ識別子への何回目の送信か。

        Returns
        -------
        dict
            JSONL に書いたものと同じ記録。

        Raises
        ------
        ValueError
            ``text`` が文字列でないか、``task_type`` が一覧に無いとき（送らない）。
        """
        if not isinstance(text, str):
            raise ValueError("embeddings の text は文字列 1 つだけ")
        if task_type not in EMBEDDING_TASK_TYPES:
            raise ValueError(f"embeddings の task_type が不正: {task_type!r}")
        body = {"model": model, "input": text, "task_type": task_type}
        return self._send(Endpoint.EMBEDDINGS, model, body, question_id=question_id, attempt=attempt)

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        task_type: str | None = None,
        query_instruction: str | None = None,
        model: str = RERANK_MODEL,
        question_id: str | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        """Rerank（``POST /rerank``）に送り、記録を返す。

        Parameters
        ----------
        query : str
            クエリ。
        documents : list of str
            並べ替える文書（1〜20 件）。
        task_type : str, optional
            ``RERANK_TASK_TYPES`` のいずれか。省略時は送らない（API 側の既定 ``GENERAL_RETRIEVAL``）。
        query_instruction : str, optional
            ``task_type`` が ``CUSTOMIZE`` のときの指示文（そのときは必須）。
        model : str, optional
            既定は ``sarashina3-rerank``。
        question_id : str, optional
            記録用の識別子。
        attempt : int, optional
            同じ識別子への何回目の送信か。

        Returns
        -------
        dict
            JSONL に書いたものと同じ記録。

        Raises
        ------
        ValueError
            文書の件数・``task_type``・``query_instruction`` が不正なとき（送らない）。
        """
        if not 1 <= len(documents) <= 20:
            raise ValueError(f"rerank の documents は 1〜20 件（指定は {len(documents)} 件）")
        if task_type is not None and task_type not in RERANK_TASK_TYPES:
            raise ValueError(f"rerank の task_type が不正: {task_type!r}")
        if task_type == "CUSTOMIZE" and not query_instruction:
            raise ValueError("rerank の task_type が CUSTOMIZE なら query_instruction が要る")
        body: dict[str, Any] = {"model": model, "query": query, "documents": list(documents)}
        if task_type is not None:
            body["task_type"] = task_type
        if query_instruction is not None:
            body["query_instruction"] = query_instruction
        return self._send(Endpoint.RERANK, model, body, question_id=question_id, attempt=attempt)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        *,
        question_id: str | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        """Translate（``POST /translate``）に送り、記録を返す。

        API は ``model`` を無視するが、窓の数え直しと済みの判定が記録の ``request.body.model``
        を見るので、本文に ``"model": "sarashina3-translate"`` を必ず入れる。

        Parameters
        ----------
        text : str
            訳す文（5,000 コードポイントまで）。
        source_lang, target_lang : str
            ``ja`` か ``en``。同じ言語は不可。
        question_id : str, optional
            記録用の識別子。
        attempt : int, optional
            同じ識別子への何回目の送信か。

        Returns
        -------
        dict
            JSONL に書いたものと同じ記録。

        Raises
        ------
        ValueError
            言語の指定か文字数が不正なとき（送らない）。
        """
        langs = {"ja", "en"}
        if source_lang not in langs or target_lang not in langs or source_lang == target_lang:
            raise ValueError(f"translate の言語は ja と en の組だけ（{source_lang}->{target_lang}）")
        if len(text) > TRANSLATE_MAX_CHARS:
            raise ValueError(f"translate の text は {TRANSLATE_MAX_CHARS} 文字まで（{len(text)} 文字）")
        body = {"model": TRANSLATE_MODEL, "text": text, "source_lang": source_lang, "target_lang": target_lang}
        return self._send(Endpoint.TRANSLATE, TRANSLATE_MODEL, body, question_id=question_id, attempt=attempt)

    def _send(
        self,
        endpoint: Endpoint,
        model: str,
        body: dict[str, Any],
        *,
        question_id: str | None,
        attempt: int,
    ) -> dict[str, Any]:
        """送り直しの規則に従って POST し、送った回ごとに記録する（各エンドポイント共通）。

        429 は 61 秒待って 1 回だけ送り直し、また 429 なら ``RateLimitExceededError``。
        500・503 は 2 秒・4 秒待って 2 回まで送り直す。それ以外は送り直さない。
        送り直しも窓の limiter を通す。最後の記録を返す。
        """
        retry_wait = 0.0
        after_429 = False
        server_retries = 0
        while True:
            record = self._send_once(
                endpoint,
                model,
                body,
                question_id=question_id,
                attempt=attempt,
                retry_wait=retry_wait,
                is_retry_after_429=after_429,
            )
            status = record["response"]["status"]
            if status == 200:
                self._completed.add((model, question_id, attempt))
                return record
            if status == 429:
                if after_429:
                    raise RateLimitExceededError(
                        f"429 が 2 回続いたので実行を止める（model={model}, question_id={question_id}）",
                        record,
                    )
                after_429, retry_wait = True, RETRY_AFTER_429_SEC
                continue
            if status in SERVER_ERROR_STATUSES and server_retries < len(SERVER_ERROR_BACKOFF_SEC):
                retry_wait = SERVER_ERROR_BACKOFF_SEC[server_retries]
                server_retries += 1
                after_429 = False
                continue
            return record

    def _send_once(
        self,
        endpoint: Endpoint,
        model: str,
        body: dict[str, Any],
        *,
        question_id: str | None,
        attempt: int,
        retry_wait: float,
        is_retry_after_429: bool,
    ) -> dict[str, Any]:
        """送り直しの待ちと窓の待ちを入れて 1 回 POST し、往復を記録する。"""
        url = self._base_url + endpoint.value
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if retry_wait > 0:
            self._clock.sleep(retry_wait)
        waited = retry_wait + self._limiter.acquire(model)

        record: dict[str, Any] = {
            "run_id": self.run_id,
            "question_id": question_id,
            "attempt": attempt,
            "wait_before_send_sec": round(waited, 3),
            "is_retry_after_429": is_retry_after_429,
            "timeout_sec": self._timeout_sec,
            "request": {"method": "POST", "url": url, "headers": _redact_headers(headers), "body": body},
        }
        req = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST"
        )
        sent = self._clock.now()
        error = None
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
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
