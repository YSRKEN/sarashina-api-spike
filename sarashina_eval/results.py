"""得点の CSV（``results/``）の名前と列の決まり。

``score.py``・``non_chat_score.py``・``tools.py`` が同じ名前の付け方を使い、
``readme_table.py`` が Chat の CSV を読むときの列もここで決める。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

#: CSV の種類。ファイル名の ``<種類>`` に入る。
KINDS: tuple[str, ...] = ("chat", "non-chat", "tools")

#: Chat の CSV の列（縦長。1 行が 1 モデル × 1 分野）。
CHAT_COLUMNS: tuple[str, ...] = ("run_id", "run_date", "model", "category", "accuracy", "answers",
                                 "all_correct", "partial", "none_correct", "ungraded")


def scores_csv_path(results_dir: Path, day: date, run_id: str, kind: str) -> Path:
    """得点の CSV のパスを返す。

    Parameters
    ----------
    results_dir : Path
        置き場（既定は ``results/``）。
    day : date
        実行の最初の送信の UTC 日付。
    run_id : str
        実行 ID の全部（先頭を切らない）。
    kind : str
        ``chat``・``non-chat``・``tools`` のどれか。

    Returns
    -------
    Path
        ``<results_dir>/<日付>-<実行 ID>-<種類>-scores.csv``。

    Raises
    ------
    ValueError
        ``kind`` が決まりの外のとき。
    """
    if kind not in KINDS:
        raise ValueError(f"kind は {KINDS} のどれか: {kind!r}")
    return results_dir / f"{day:%Y-%m-%d}-{run_id}-{kind}-scores.csv"
