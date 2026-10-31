"""モデルごとのレート上限とエンドポイントの定義。

レート上限は公式 https://docs.sarashina.ai/rate-limits による（固定 1 分窓。
レート関連のヘッダーも Retry-After も返らない）。
"""

from __future__ import annotations

from enum import Enum

BASE_URL = "https://api.sarashina.ai/v1"

#: モデル名ごとの 1 分あたりの上限回数（RPM）。2026-10-31 に公式の値を確認した。
MODEL_RPM: dict[str, int] = {
    "sarashina3.2-mini-thinking": 30,
    "sarashina3-mini": 30,
    "sarashina3-guard": 6,
    "sarashina3-translate": 30,
    "sarashina3-nano": 60,
    "sarashina3-embedding": 60,
    "sarashina3-rerank": 60,
}

THINKING_MODEL = "sarashina3.2-mini-thinking"
#: ``reasoning_effort`` を受け付けるモデル（ほかのモデルに付けると 400）。
THINKING_MODELS: frozenset[str] = frozenset({THINKING_MODEL})
#: ``reasoning_effort`` の値（https://docs.sarashina.ai/api-reference/chat-completions 、2026-10-31 に確認）。
#: 既定は公式に記載が無いので、指定しないときは本文に入れない。
REASONING_EFFORTS: tuple[str, ...] = ("low", "medium", "high")


def reasoning_params(models: list[str] | tuple[str, ...], effort: str | None) -> dict[str, str]:
    """送り先と推論の強さを確かめ、本文に足す項目を返す。

    Parameters
    ----------
    models : list or tuple of str
        送り先のモデル。
    effort : str or None
        ``REASONING_EFFORTS`` のいずれか。None なら指定しない。

    Returns
    -------
    dict
        ``{"reasoning_effort": effort}``。指定しないなら空。

    Raises
    ------
    ValueError
        知らない値か、``THINKING_MODELS`` に無いモデルへ渡そうとしたとき（送らない）。
    """
    if effort is None:
        return {}
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort は {', '.join(REASONING_EFFORTS)} のいずれか（指定は {effort}）")
    others = [m for m in models if m not in THINKING_MODELS]
    if others:
        raise ValueError(f"reasoning_effort を受け付けないモデル: {', '.join(others)}")
    return {"reasoning_effort": effort}


GUARD_MODEL = "sarashina3-guard"
EMBEDDING_MODEL = "sarashina3-embedding"
RERANK_MODEL = "sarashina3-rerank"
TRANSLATE_MODEL = "sarashina3-translate"

#: Embeddings の ``task_type`` の値（https://docs.sarashina.ai/api-reference/embeddings 、2026-10-13 に確認）。
EMBEDDING_TASK_TYPES: frozenset[str] = frozenset(
    {
        "RETRIEVAL_DOCUMENT",
        "SEMANTIC_SIMILARITY",
        "RETRIEVAL_QUERY",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
        "CLASSIFICATION",
        "CLUSTERING",
    }
)

#: Rerank の ``task_type`` の値（https://docs.sarashina.ai/api-reference/rerank 、2026-10-13 に確認）。
RERANK_TASK_TYPES: frozenset[str] = frozenset(
    {"GENERAL_RETRIEVAL", "GENERAL_QA", "NEWS_RETRIEVAL", "SEMANTIC_SIMILARITY", "CUSTOMIZE"}
)

#: Translate の ``text`` の上限（Unicode コードポイント数）。
TRANSLATE_MAX_CHARS = 5000


class Endpoint(str, Enum):
    """API のエンドポイントのパス（``BASE_URL`` からの相対）。

    パスは公式の API リファレンス（2026-10-13 に確認）による。
    Translate は ``model`` の指定が不要で、渡しても無視される（sarashina3-translate が使われる）。
    """

    CHAT = "/chat/completions"
    EMBEDDINGS = "/embeddings"
    RERANK = "/rerank"
    TRANSLATE = "/translate"
