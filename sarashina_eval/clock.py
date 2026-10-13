"""現在時刻と待機を差し替えられるようにする時計。"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """現在時刻（UNIX 秒）を返し、指定の秒数だけ待つもの。"""

    def now(self) -> float:
        """現在時刻を UNIX 秒で返す。"""
        ...

    def sleep(self, seconds: float) -> None:
        """``seconds`` 秒だけ待つ。"""
        ...


class SystemClock:
    """実際の時計。``time.time`` と ``time.sleep`` をそのまま使う。"""

    def now(self) -> float:
        """現在時刻を UNIX 秒で返す。

        Returns
        -------
        float
            ``time.time()`` の値。
        """
        return time.time()

    def sleep(self, seconds: float) -> None:
        """``seconds`` 秒だけ実際に待つ。

        Parameters
        ----------
        seconds : float
            待つ秒数。0 以下なら待たない。
        """
        if seconds > 0:
            time.sleep(seconds)
