"""問題集（分野ごとの JSONL）の 1 行の形と、その読み込み・検査。

1 行 1 問の JSON で、項目は次のとおり。

- ``id`` (str): 問題集全体で一意な識別子。例 ``"math-001"``
- ``category`` (str): 分野。``CATEGORIES`` のいずれか
- ``prompt`` (str): モデルに送る設問の本文
- ``expected`` (str | None): 期待する答え。人の判定だけの問題では ``null`` でよい
- ``grading`` (object): 採点方法。``{"method": <GRADING_METHODS のいずれか>, ...}``。
  方法ごとの追加の項目（``tolerance``・``pattern``・``tests`` など）はそのまま保持する
- ``intent`` (str): 問題の狙い（何を見るための問題か）
- ``system`` (str, 任意): system メッセージ。無ければ送らない
- ``tags`` (list[str], 任意): 絞り込み用の札
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 知っている分野。
CATEGORIES: frozenset[str] = frozenset({"logic", "math", "knowledge", "japanese", "instruction", "code"})

#: 知っている採点方法。各方法の追加の項目は課題 04・06・07 で検査する。
#: exact: 完全一致 / numeric: 数値の一致（``tolerance``）/ regex: 正規表現（``pattern``）/
#: code_tests: コードのテスト（``tests``: 入力と期待値の組の並び）/ human: 人の判定
GRADING_METHODS: frozenset[str] = frozenset({"exact", "numeric", "regex", "code_tests", "human"})

_REQUIRED_STR = ("id", "category", "prompt", "intent")


class QuestionSetError(ValueError):
    """問題集に壊れた行があるときの例外。``problems`` に全件を持つ。

    Parameters
    ----------
    problems : list of str
        ``"<ファイル>:<行番号>: <理由>"`` の並び。
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("問題集に壊れた行がある:\n" + "\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Grading:
    """採点方法。

    Attributes
    ----------
    method : str
        ``GRADING_METHODS`` のいずれか。
    params : dict
        方法ごとの追加の項目（``method`` 以外の全項目）。
    """

    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Question:
    """問題集の 1 問。項目の意味はモジュールの docstring を参照。"""

    id: str
    category: str
    prompt: str
    expected: str | None
    grading: Grading
    intent: str
    system: str | None = None
    tags: tuple[str, ...] = ()

    def messages(self) -> list[dict[str, str]]:
        """Chat に送る ``messages`` を作る。

        Returns
        -------
        list of dict
            ``system`` があれば先頭に置き、続けて ``prompt`` を user として置く。
        """
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        msgs.append({"role": "user", "content": self.prompt})
        return msgs


def parse_question(obj: Any) -> Question:
    """JSON から読んだ 1 行を ``Question`` にする。

    Parameters
    ----------
    obj : Any
        ``json.loads`` の結果。

    Returns
    -------
    Question

    Raises
    ------
    ValueError
        形が合わないとき。理由を日本語で持つ。
    """
    if not isinstance(obj, dict):
        raise ValueError("1 行が JSON のオブジェクトではない")
    for key in _REQUIRED_STR:
        if key not in obj:
            raise ValueError(f"必須の項目 {key!r} が無い")
        if not isinstance(obj[key], str) or not obj[key].strip():
            raise ValueError(f"項目 {key!r} が空でない文字列ではない")
    if "expected" not in obj:
        raise ValueError("必須の項目 'expected' が無い（無いなら null を書く）")
    if obj["expected"] is not None and not isinstance(obj["expected"], str):
        raise ValueError("項目 'expected' が文字列か null ではない")
    if obj["category"] not in CATEGORIES:
        raise ValueError(f"分野 {obj['category']!r} は知らない値（{', '.join(sorted(CATEGORIES))}）")
    g = obj.get("grading")
    if not isinstance(g, dict) or "method" not in g:
        raise ValueError("必須の項目 'grading'（method を持つオブジェクト）が無い")
    if g["method"] not in GRADING_METHODS:
        raise ValueError(f"採点方法 {g['method']!r} は知らない値（{', '.join(sorted(GRADING_METHODS))}）")
    system = obj.get("system")
    if system is not None and not isinstance(system, str):
        raise ValueError("項目 'system' が文字列ではない")
    tags = obj.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ValueError("項目 'tags' が文字列の配列ではない")
    return Question(
        id=obj["id"],
        category=obj["category"],
        prompt=obj["prompt"],
        expected=obj["expected"],
        grading=Grading(g["method"], {k: v for k, v in g.items() if k != "method"}),
        intent=obj["intent"],
        system=system,
        tags=tuple(tags),
    )


def load_questions(paths: list[Path]) -> list[Question]:
    """JSONL の問題集を読む。ディレクトリなら直下の ``*.jsonl`` を名前順に読む。

    壊れた行が 1 件でもあれば何も返さず、全件を行番号つきで ``QuestionSetError`` にまとめる。
    空行は飛ばす。ID の重複も壊れた行として扱う。

    Parameters
    ----------
    paths : list of Path
        ファイルかディレクトリの並び。

    Returns
    -------
    list of Question
        読んだ順の問題。

    Raises
    ------
    QuestionSetError
        壊れた行があるとき。
    """
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    questions: list[Question] = []
    problems: list[str] = []
    seen: dict[str, str] = {}
    for f in files:
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{f}:{lineno}"
            try:
                q = parse_question(json.loads(line))
            except json.JSONDecodeError as e:
                problems.append(f"{where}: JSON として読めない（{e.msg}）")
                continue
            except ValueError as e:
                problems.append(f"{where}: {e}")
                continue
            if q.id in seen:
                problems.append(f"{where}: ID {q.id!r} が {seen[q.id]} と重複")
                continue
            seen[q.id] = where
            questions.append(q)
    if problems:
        raise QuestionSetError(problems)
    return questions
