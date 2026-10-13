"""Sarashina API に最小のリクエストを 1 回だけ送り、HTTP の往復をすべて記録する。

送るのは sarashina3-nano への 1 回だけで、再送はしない。送信と記録は
``sarashina_eval.SarashinaClient`` が行う。

使い方
------
    python3 scripts/first_call.py

API キーは環境変数か、リポジトリ直下の ``.env`` の ``SARASHINA_API_KEY`` から読む。
記録は ``logs/YYYY-MM-DD.jsonl`` に 1 行で追記する（``logs/`` は公開しない）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sarashina_eval import ApiKeyMissingError, SarashinaClient  # noqa: E402


def main() -> int:
    """1 回送って結果を表示する。

    Returns
    -------
    int
        200 なら 0、キーが空なら 2、それ以外は 1。
    """
    try:
        client = SarashinaClient()
    except ApiKeyMissingError as e:
        print(e, file=sys.stderr)
        return 2
    rec = client.chat(
        "sarashina3-nano",
        [{"role": "user", "content": "こんにちは。一言で返事をしてください。"}],
        max_completion_tokens=32,
    )
    status = rec["response"]["status"]
    print(f"status={status} elapsed_ms={rec['elapsed_ms']}")
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
