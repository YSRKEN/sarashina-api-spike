"""得点の CSV ごとに、採点の記録が公開リポジトリに入っていることを確かめる。

2026-10-13 の記録は Git の外に置いたため、流したマシンにしか残らず、採点し直せなくなった。
同じことを繰り返さないよう、CSV だけをコミットして記録を置き忘れたら、ここで落とす。
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: ``<日付>-<実行 ID>-<種類>-scores.csv`` から実行 ID を取る。
CSV_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-(?P<run>.+?)-(?:chat|non-chat|tools)-scores\.csv$")


class ResultsHaveLogsTest(unittest.TestCase):
    def test_every_scores_csv_has_committable_scores_log(self) -> None:
        csvs = sorted((ROOT / "results").rglob("*-scores.csv"))
        self.assertTrue(csvs)
        missing, ignored = [], []
        for csv_path in csvs:
            run = CSV_NAME.match(csv_path.name)["run"]
            logs = sorted((ROOT / "logs" / "scores").glob(f"{run}*.jsonl"))
            if not logs:
                missing.append(csv_path.name)
                continue
            r = subprocess.run(["git", "check-ignore", "-q", str(logs[0])], cwd=ROOT)
            if r.returncode == 0:
                ignored.append(logs[0].name)
        self.assertEqual(missing, [], "採点の記録が logs/scores/ に無い")
        self.assertEqual(ignored, [], "採点の記録が .gitignore で Git の外になっている")


if __name__ == "__main__":
    unittest.main()
