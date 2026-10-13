"""モデルごとに直近 60 秒の送信時刻を数え、上限に達したら待たせる。"""

from __future__ import annotations

from collections import deque

from sarashina_eval.clock import Clock

WINDOW_SEC = 60.0
#: 上限に達したとき、一番古い送信からこの秒数がたつまで待つ（窓に 1 秒の余裕を足す）。
WAIT_FROM_OLDEST_SEC = 61.0


class RateLimiter:
    """モデルごとの送信時刻の窓を持ち、送る前に必要な待ちを入れる。

    Parameters
    ----------
    rpm_by_model : dict[str, int]
        モデル名ごとの 1 分あたりの上限回数。
    clock : Clock
        現在時刻と待機の取り出し口。
    """

    def __init__(self, rpm_by_model: dict[str, int], clock: Clock) -> None:
        self._rpm = dict(rpm_by_model)
        self._clock = clock
        self._sent: dict[str, deque[float]] = {}

    def acquire(self, model: str) -> float:
        """``model`` に 1 回送ってよくなるまで待ち、送信時刻を記録する。

        Parameters
        ----------
        model : str
            送り先のモデル名。``rpm_by_model`` に無ければ ``KeyError``。

        Returns
        -------
        float
            窓のために待った秒数（待たなければ 0.0）。
        """
        rpm = self._rpm[model]
        sent = self._sent.setdefault(model, deque())
        waited = 0.0
        now = self._clock.now()
        self._prune(sent, now)
        if len(sent) >= rpm:
            wait = sent[0] + WAIT_FROM_OLDEST_SEC - now
            if wait > 0:
                self._clock.sleep(wait)
                waited = wait
            now = self._clock.now()
            self._prune(sent, now)
        sent.append(now)
        return waited

    @staticmethod
    def _prune(sent: deque[float], now: float) -> None:
        """窓の外（60 秒以上前）の送信時刻を捨てる。"""
        while sent and sent[0] <= now - WINDOW_SEC:
            sent.popleft()
