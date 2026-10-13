"""Tool Calling の検証に使う偽の道具。本物の通信はせず、決まった結果を返す。

道具の定義（OpenAI 形式の ``tools`` の要素）と、呼び出しを実行する ``execute`` を持つ。
結果はいつも JSON の文字列で、失敗は ``{"error": ...}`` で返す（例外にしない）。
"""

from __future__ import annotations

import ast
import json
import operator
from typing import Any, Callable

#: web_fetch が知っている架空の URL と本文。
PAGES: dict[str, str] = {
    "https://example.test/kotori-bakery/news": (
        "ことりベーカリー お知らせ（2026年10月）\n"
        "・10月20日（火）は設備点検のため臨時休業します。\n"
        "・10月から新商品「栗とほうじ茶のデニッシュ」を1個320円で販売します。\n"
        "・営業時間は朝7時から夕方6時までです。毎週水曜日が定休日です。"
    ),
    "https://example.test/news/aozora-library": (
        "あおぞら市立図書館、2027年4月に移転へ\n"
        "あおぞら市は、市立図書館を駅前の新しい複合施設「みなもプラザ」の3階へ移すと発表した。"
        "蔵書は約18万冊で、移転に伴い2027年3月の1か月間は休館する。"
        "新館には自習席が120席設けられる。"
    ),
}

#: get_weather が知っている架空の都市と日付ごとの天気。
WEATHER: dict[str, dict[str, dict[str, Any]]] = {
    "ほしぞら市": {
        "2026-10-14": {"weather": "晴れ", "max_c": 22, "min_c": 13, "precip_pct": 10},
        "2026-10-15": {"weather": "雨", "max_c": 17, "min_c": 12, "precip_pct": 80},
    },
    "かざみ町": {
        "2026-10-14": {"weather": "くもり", "max_c": 19, "min_c": 9, "precip_pct": 40},
        "2026-10-15": {"weather": "晴れ", "max_c": 21, "min_c": 10, "precip_pct": 0},
    },
}

#: 道具の定義（``tools`` にそのまま渡す形）。
TOOL_DEFS: dict[str, dict[str, Any]] = {
    "web_fetch": {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "指定した URL のページを取得し、本文のテキストを返す。",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "取得するページの URL"}},
                "required": ["url"],
            },
        },
    },
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "都市と日付を指定して、天気・最高気温・最低気温（摂氏）・降水確率（%）を返す。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "都市名（例: ほしぞら市）"},
                    "date": {"type": "string", "description": "日付（YYYY-MM-DD）"},
                },
                "required": ["city", "date"],
            },
        },
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "四則演算の式を計算して結果を返す（+ - * / と括弧）。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "計算する式（例: 320*3）"}},
                "required": ["expression"],
            },
        },
    },
}

_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    """四則演算だけの式の木を評価する。それ以外は ValueError。"""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("四則演算以外は使えない")


def web_fetch(url: str) -> dict[str, Any]:
    """架空のページを返す。知らない URL は 404 相当のエラー。"""
    if url in PAGES:
        return {"status": 200, "url": url, "text": PAGES[url]}
    return {"error": "404 Not Found", "url": url}


def get_weather(city: str, date: str) -> dict[str, Any]:
    """架空の天気を返す。知らない都市・日付はエラー。"""
    if city not in WEATHER:
        return {"error": f"都市が見つからない: {city}"}
    if date not in WEATHER[city]:
        return {"error": f"この日付の予報は無い: {date}", "available_dates": sorted(WEATHER[city])}
    return {"city": city, "date": date, **WEATHER[city][date]}


def calculator(expression: str) -> dict[str, Any]:
    """四則演算の式を計算する。"""
    try:
        value = _eval(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return {"error": f"計算できない: {e}"}
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"expression": expression, "result": value}


_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "web_fetch": web_fetch, "get_weather": get_weather, "calculator": calculator,
}


def tool_definitions(names: list[str]) -> list[dict[str, Any]]:
    """名前の並びから ``tools`` に渡す定義の並びを作る。

    Parameters
    ----------
    names : list of str

    Returns
    -------
    list of dict

    Raises
    ------
    KeyError
        知らない道具の名前のとき。
    """
    return [TOOL_DEFS[n] for n in names]


def execute(name: str, arguments: str) -> str:
    """モデルが要求した呼び出しを実行し、tool メッセージの本文（JSON 文字列）を返す。

    Parameters
    ----------
    name : str
        関数名。
    arguments : str
        引数の JSON 文字列（``tool_calls[].function.arguments``）。

    Returns
    -------
    str
        結果の JSON。知らない道具・壊れた引数・足りない引数も ``{"error": ...}`` で返す。
    """
    impl = _IMPLS.get(name)
    if impl is None:
        out: dict[str, Any] = {"error": f"知らない道具: {name}"}
    else:
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            args, out = None, {"error": f"引数が JSON として読めない: {e}"}
        if args is not None:
            if not isinstance(args, dict):
                out = {"error": "引数は JSON のオブジェクトであること"}
            else:
                try:
                    out = impl(**{k: str(v) for k, v in args.items()})
                except TypeError as e:
                    out = {"error": f"引数が合わない: {e}"}
    return json.dumps(out, ensure_ascii=False)
