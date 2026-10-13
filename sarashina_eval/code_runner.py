"""コード分野の答えを Docker の ``python:3`` で実行して採点する。

答えの本文からコードブロックを抜き出し、問題の ``grading`` が持つテスト（関数名と、引数と期待値の組）と
1 つのスクリプトにまとめ、標準入力で ``docker run`` に渡す。結果は標準出力の JSON で受け取る。

``grading`` の形::

    {"method": "code_tests", "function": "関数名",
     "tests": [{"input": [引数1, 引数2, ...], "output": 期待値}, ...]}

戻り値は JSON に直してから（タプルは配列になる）期待値と ``==`` で比べる。
"""

from __future__ import annotations

import json
import re
import secrets
import subprocess
from dataclasses import dataclass
from typing import Any

#: 実行に使うイメージ。
IMAGE = "python:3"

#: 1 回の実行の時間制限（秒）。
TIMEOUT_SEC = 10.0

#: コンテナの制限。ネットワークなし・読み取り専用・メモリ 256MB・CPU 1・プロセス数 64。
DOCKER_LIMITS: tuple[str, ...] = (
    "--network", "none", "--read-only", "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
)

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)

_HARNESS = r'''
import json, sys, io, contextlib
_MARK = {mark!r}
_out = sys.stdout
_res = {{"passed": 0, "total": 0, "results": [], "error": None}}
_tests = json.loads({tests!r})
_ns = {{"__name__": "__answer__"}}
try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile({code!r}, "<answer>", "exec"), _ns)
    _fn = _ns.get({function!r})
    if not callable(_fn):
        raise NameError("関数 " + {function!r} + " が定義されていない")
except BaseException as e:
    _res["error"] = "load: " + type(e).__name__ + ": " + str(e)[:200]
    _fn = None
_res["total"] = len(_tests)
if _fn is not None:
    for _t in _tests:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _got = _fn(*_t["input"])
            _ok = json.loads(json.dumps(_got)) == _t["output"]
            _res["results"].append({{"ok": _ok}})
        except BaseException as e:
            _ok = False
            _res["results"].append({{"ok": False, "error": type(e).__name__ + ": " + str(e)[:200]}})
        _res["passed"] += int(_ok)
_out.write("\n" + _MARK + json.dumps(_res) + "\n")
_out.flush()
'''


@dataclass(frozen=True)
class CodeResult:
    """1 つの答えを実行した結果。

    Attributes
    ----------
    passed : int
        通ったテストの件数。
    total : int
        テストの件数。
    error : str or None
        実行全体の失敗（時間切れ・読み込みの失敗・結果が読めない）の理由。無ければ None。
    """

    passed: int
    total: int
    error: str | None = None

    @property
    def all_passed(self) -> bool:
        """全件通ったか（テストが 1 件以上あること）。"""
        return self.total > 0 and self.passed == self.total


def extract_code(content: str | None) -> str | None:
    """本文からコードブロックを抜き出す。

    ``python``（または ``py``・言語名なし）のブロックがあれば、それらを順につなぐ。
    無ければ None。

    Parameters
    ----------
    content : str or None

    Returns
    -------
    str or None
    """
    if not isinstance(content, str):
        return None
    blocks = [body for lang, body in _FENCE.findall(content) if lang.lower() in ("", "python", "py", "python3")]
    code = "\n\n".join(b.strip("\n") for b in blocks)
    return code if code.strip() else None


def validate_params(params: dict[str, Any]) -> None:
    """``code_tests`` の ``grading`` の追加の項目を検査する。

    Parameters
    ----------
    params : dict

    Raises
    ------
    ValueError
        ``function`` が識別子でない、``tests`` が空、各テストに ``input``（配列）・``output`` が無いとき。
    """
    fn = params.get("function")
    if not isinstance(fn, str) or not fn.isidentifier():
        raise ValueError("code_tests の 'function' が識別子の文字列ではない")
    tests = params.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError("code_tests の 'tests' が空でない配列ではない")
    for i, t in enumerate(tests):
        if not isinstance(t, dict) or not isinstance(t.get("input"), list) or "output" not in t:
            raise ValueError(f"code_tests の tests[{i}] に 'input'（配列）と 'output' が無い")


def build_script(code: str, function: str, tests: list[dict[str, Any]], mark: str) -> str:
    """答えのコードとテストを 1 つのスクリプトにまとめる。

    結果は ``mark`` で始まる行に JSON で出す。答えの ``print`` は捨てる。

    Parameters
    ----------
    code : str
    function : str
    tests : list of dict
    mark : str
        結果の行の目印。答えが真似できないよう実行ごとに乱数で作る。

    Returns
    -------
    str
    """
    return _HARNESS.format(mark=mark, tests=json.dumps(tests), code=code, function=function)


def docker_available(timeout: float = 10.0) -> bool:
    """Docker が使えるか（``docker info`` が成功するか）。

    Returns
    -------
    bool
    """
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def run_tests(code: str, params: dict[str, Any], timeout: float = TIMEOUT_SEC) -> CodeResult:
    """答えのコードを Docker で実行し、テストを当てる。

    Parameters
    ----------
    code : str
        抜き出したコード。
    params : dict
        ``validate_params`` を通る ``grading`` の追加の項目。
    timeout : float
        秒。超えたらコンテナを止め、全件失敗とする。

    Returns
    -------
    CodeResult
    """
    validate_params(params)
    tests = params["tests"]
    mark = "__RESULT_" + secrets.token_hex(16) + "__"
    name = "sarashina-eval-" + secrets.token_hex(6)
    cmd = ["docker", "run", "--rm", "-i", "--name", name, *DOCKER_LIMITS, IMAGE, "python", "-I", "-"]
    script = build_script(code, params["function"], tests, mark)
    try:
        r = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
        return CodeResult(0, len(tests), f"timeout: {timeout:g} 秒を超えた")
    for line in reversed(r.stdout.splitlines()):
        if line.startswith(mark):
            res = json.loads(line[len(mark):])
            return CodeResult(int(res["passed"]), int(res["total"]), res.get("error"))
    tail = (r.stderr or "").strip().splitlines()[-1:] or [f"exit {r.returncode}"]
    return CodeResult(0, len(tests), "no_result: " + tail[0][:200])
