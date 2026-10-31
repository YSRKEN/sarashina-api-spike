"""実行の入口が ``SarashinaClient`` に渡す引数を組み立てる。"""

from __future__ import annotations


def timeout_kwargs(timeout: float | None) -> dict[str, float]:
    """``--timeout`` を ``SarashinaClient`` の引数にする。

    Parameters
    ----------
    timeout : float or None
        秒数。None なら既定（60 秒）に任せる。

    Returns
    -------
    dict
    """
    return {} if timeout is None else {"timeout_sec": timeout}
