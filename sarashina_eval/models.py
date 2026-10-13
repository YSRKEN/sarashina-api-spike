"""モデルごとのレート上限とエンドポイントの定義。

レート上限は公式 https://docs.sarashina.ai/rate-limits による（固定 1 分窓。
レート関連のヘッダーも Retry-After も返らない）。
"""

from __future__ import annotations

from enum import Enum

BASE_URL = "https://api.sarashina.ai/v1"

#: モデル名ごとの 1 分あたりの上限回数（RPM）。
MODEL_RPM: dict[str, int] = {
    "sarashina3-mini": 6,
    "sarashina3-guard": 6,
    "sarashina3-translate": 6,
    "sarashina3-nano": 60,
    "sarashina3-embedding": 60,
    "sarashina3-rerank": 60,
}


class Endpoint(str, Enum):
    """API のエンドポイントのパス（``BASE_URL`` からの相対）。

    今回使うのは ``CHAT`` だけ。ほかは後で足すときの置き場として持つ。
    パスは公式の文書で確かめてから使うこと。
    """

    CHAT = "/chat/completions"
    EMBEDDINGS = "/embeddings"
    RERANK = "/rerank"
    TRANSLATE = "/translate"
