"""問題集を複数のモデルへ流す。送るリクエストの計画と所要時間の見積もりもここで作る。"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from sarashina_eval.models import MODEL_RPM
from sarashina_eval.questions import Question

#: 既定で流すモデル。
DEFAULT_MODELS: tuple[str, ...] = ("sarashina3-mini", "sarashina3-nano")
#: 1 リクエストで受け取る候補数の既定。
DEFAULT_N = 3
#: 固定 1 分窓で、窓が明けるまでに待つ秒数（rate_limiter に合わせ 61 秒で見積もる）。
WINDOW_SEC = 61

#: 送る前に「済んでいるか」を問い合わせる口。``(model, question_id) -> bool`` で、真なら送らない。
#: 再開時に済んだ問題を飛ばすための差し込み口で、中身は client 側の担当が用意する。
SkipCheck = Callable[[str, str], bool]


class ChatSender(Protocol):
    """``SarashinaClient.chat`` と同じ呼び口を持つもの。"""

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        question_id: str | None = None,
        attempt: int = 1,
        **params: Any,
    ) -> dict[str, Any]:
        """1 回送って記録を返す。"""
        ...


@dataclass(frozen=True)
class PlannedRequest:
    """送る予定の 1 リクエスト。

    Attributes
    ----------
    model : str
        送り先のモデル。
    question : Question
        設問。
    params : dict
        本文に足す項目（``n`` など）。temperature は入れない（既定 0.7 に任せる）。
    """

    model: str
    question: Question
    params: dict[str, Any] = field(default_factory=dict)


def plan_requests(
    questions: list[Question], models: tuple[str, ...] = DEFAULT_MODELS, n: int = DEFAULT_N
) -> list[PlannedRequest]:
    """送る順にリクエストを並べる。1 問ごとに各モデルへ 1 回ずつ送る。

    Parameters
    ----------
    questions : list of Question
    models : tuple of str
        送り先。``MODEL_RPM`` にあるもの。
    n : int
        1 リクエストで受け取る候補数（1〜5）。

    Returns
    -------
    list of PlannedRequest

    Raises
    ------
    ValueError
        ``n`` が範囲外か、知らないモデルのとき。
    """
    if not 1 <= n <= 5:
        raise ValueError(f"n は 1〜5（指定は {n}）")
    unknown = [m for m in models if m not in MODEL_RPM]
    if unknown:
        raise ValueError(f"知らないモデル: {', '.join(unknown)}")
    return [PlannedRequest(m, q, {"n": n}) for q in questions for m in models]


def estimate_seconds(
    plan: list[PlannedRequest], rpm_by_model: dict[str, int] | None = None
) -> dict[str, int]:
    """モデルごとの窓待ちの合計秒数を見積もる（応答の時間は含めない）。

    固定 1 分窓なので、``k`` 回送るモデルは ``ceil(k / rpm) - 1`` 回、窓の明けを待つ。

    Parameters
    ----------
    plan : list of PlannedRequest
    rpm_by_model : dict, optional
        既定は ``MODEL_RPM``。

    Returns
    -------
    dict
        モデル名から見積もり秒数への対応。
    """
    rpm = rpm_by_model or MODEL_RPM
    counts = Counter(r.model for r in plan)
    return {m: max(math.ceil(k / rpm[m]) - 1, 0) * WINDOW_SEC for m, k in counts.items()}


def format_dry_run(plan: list[PlannedRequest], rpm_by_model: dict[str, int] | None = None) -> str:
    """``--dry-run`` で出す、リクエストの一覧と所要時間の見積もりの文を作る。

    Parameters
    ----------
    plan : list of PlannedRequest
    rpm_by_model : dict, optional
        既定は ``MODEL_RPM``。

    Returns
    -------
    str
    """
    rpm = rpm_by_model or MODEL_RPM
    lines = [f"送るリクエスト {len(plan)} 件（実際には送らない）:"]
    for i, r in enumerate(plan, start=1):
        params = " ".join(f"{k}={v}" for k, v in r.params.items())
        lines.append(f"  {i:>3}. {r.model:<18} {r.question.id:<14} [{r.question.category}] {params}")
    est = estimate_seconds(plan, rpm)
    counts = Counter(r.model for r in plan)
    lines.append("所要時間の見積もり（窓待ちのみ。応答の時間は含めない）:")
    for m, sec in est.items():
        lines.append(f"  {m:<18} {counts[m]} 件 / {rpm[m]} RPM -> 約 {sec} 秒")
    lines.append(f"  全体（最も待つモデルで決まる） -> 約 {max(est.values(), default=0)} 秒")
    return "\n".join(lines)


def run_plan(
    client: ChatSender, plan: list[PlannedRequest], should_skip: SkipCheck | None = None
) -> list[dict[str, Any]]:
    """計画どおりに送る。記録（ログへの追記）は client が行う。

    Parameters
    ----------
    client : ChatSender
        ``SarashinaClient`` など。
    plan : list of PlannedRequest
    should_skip : callable, optional
        ``(model, question_id) -> bool``。真なら送らない（再開用の差し込み口）。

    Returns
    -------
    list of dict
        送ったリクエストの記録。
    """
    records = []
    for r in plan:
        if should_skip is not None and should_skip(r.model, r.question.id):
            continue
        records.append(client.chat(r.model, r.question.messages(), question_id=r.question.id, **r.params))
    return records
