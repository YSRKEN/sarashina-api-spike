"""実行ごとに、送った問題集・入力の写しを ``<log-dir>/runs/<run_id>/`` に残し、採点で読み戻す。

置き場の形::

    <log-dir>/runs/<run_id>/manifest.json
    <log-dir>/runs/<run_id>/questions/math.jsonl        # 元は questions/math.jsonl
    <log-dir>/runs/<run_id>/inputs/guard.jsonl          # 元は inputs/guard.jsonl
    <log-dir>/runs/<run_id>/inputs_private/guard.jsonl  # 元は inputs_private/guard.jsonl

写しのファイルは ``<元の親ディレクトリ名>/<元のファイル名>``。親の名前が重なるときは
``<名前>_2`` のように番号を足す。``manifest.json`` の ``files`` に、写しの相対パス・元の絶対パス・
種類（``questions`` か ``inputs``）を並べる。行は元のファイルの文字列をそのまま写す。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: 写しの置き場（``log_dir`` の下）。
RUNS_DIR = "runs"
#: 写しの目録のファイル名。
MANIFEST = "manifest.json"


def snapshot_dir(log_dir: Path, run_id: str) -> Path:
    """``run_id`` の写しの置き場を返す。

    Parameters
    ----------
    log_dir : Path
    run_id : str

    Returns
    -------
    Path
        ``<log_dir>/runs/<run_id>``（存在するとは限らない）。
    """
    return log_dir / RUNS_DIR / run_id


def _line_id(line: str) -> str | None:
    """JSONL の 1 行から ``id`` を取り出す。読めなければ None。"""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    cid = obj.get("id") if isinstance(obj, dict) else None
    return cid if isinstance(cid, str) else None


def _select(path: Path, ids: set[str]) -> list[str]:
    """``path`` の行のうち ``id`` が ``ids`` にあるものを、元の文字列のまま返す。"""
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and _line_id(l) in ids]


def question_files(paths: Iterable[Path]) -> list[Path]:
    """``load_questions`` と同じ規則で、読む JSONL を並べる。

    Parameters
    ----------
    paths : iterable of Path
        ファイルかディレクトリ。

    Returns
    -------
    list of Path
    """
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    return files


def save_snapshot(
    log_dir: Path,
    run_id: str,
    *,
    questions: tuple[Iterable[Path], set[str]] | None = None,
    inputs: tuple[Iterable[Path], dict[str, set[str]]] | None = None,
) -> Path:
    """送る対象になった行の写しを保存する。

    既に写しがあれば上書きしない。中身が今回のものと違えば標準エラーに警告を出す。

    Parameters
    ----------
    log_dir : Path
    run_id : str
    questions : tuple, optional
        ``(問題集のパスの並び, 送る問題の ID の集合)``。
    inputs : tuple, optional
        ``(入力の置き場の並び, {種類: 送る ID の集合})``。ファイルは ``<置き場>/<種類>.jsonl``。

    Returns
    -------
    Path
        写しの置き場。
    """
    picked: list[tuple[str, Path, list[str]]] = []
    if questions is not None:
        paths, ids = questions
        for f in question_files(paths):
            if f.is_file():
                lines = _select(f, ids)
                if lines:
                    picked.append(("questions", f, lines))
    if inputs is not None:
        dirs, by_kind = inputs
        for kind, ids in by_kind.items():
            for d in dirs:
                f = d / f"{kind}.jsonl"
                if f.is_file():
                    lines = _select(f, ids)
                    if lines:
                        picked.append(("inputs", f, lines))
    files: list[dict[str, Any]] = []
    contents: dict[str, str] = {}
    folder_of: dict[Path, str] = {}
    for kind, src, lines in picked:
        parent = src.resolve().parent
        if parent not in folder_of:
            name, n = parent.name or "root", 1
            while name in folder_of.values():
                n += 1
                name = f"{parent.name}_{n}"
            folder_of[parent] = name
        rel = f"{folder_of[parent]}/{src.name}"
        files.append({"path": rel, "source": str(src.resolve()), "kind": kind})
        contents[rel] = "\n".join(lines) + "\n"
    manifest = {"run_id": run_id, "files": files}
    out = snapshot_dir(log_dir, run_id)
    if (out / MANIFEST).is_file():
        if _differs(out, manifest, contents):
            print(f"警告: 写し {out} は今回の問題集・入力と中身が違う（上書きしない）", file=sys.stderr)
        return out
    for rel, text in contents.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text, encoding="utf-8")
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def _differs(out: Path, manifest: dict[str, Any], contents: dict[str, str]) -> bool:
    """既存の写しと今回の中身が違うか。"""
    try:
        old = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if old.get("files") != manifest["files"]:
        return True
    for rel, text in contents.items():
        p = out / rel
        if not p.is_file() or p.read_text(encoding="utf-8") != text:
            return True
    return False


def snapshot_paths(log_dir: Path, run_id: str, kind: str) -> list[Path] | None:
    """写しのうち ``kind`` のファイルを返す。写しが無ければ None。

    Parameters
    ----------
    log_dir : Path
    run_id : str
    kind : str
        ``questions`` か ``inputs``。

    Returns
    -------
    list of Path or None
        写しにその種類のファイルが無いときも None。``kind`` が ``inputs`` のときは、写しの中の置き場（ディレクトリ）の並び。
    """
    out = snapshot_dir(log_dir, run_id)
    try:
        manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    paths = [out / f["path"] for f in manifest.get("files", []) if f.get("kind") == kind]
    if not paths:
        return None
    if kind == "inputs":
        return list(dict.fromkeys(p.parent for p in paths))
    return paths
