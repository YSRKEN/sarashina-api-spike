# Sarashina API 検証

Sarashina API の Chat の 2 モデル（sarashina3-mini・nano）と Chat 以外の 4 モデルを、自作の問題集 61 問と 38 件の入力で試し、どの用途に使えるか・どこで詰まるかをまとめたリポジトリ。公式のベンチマークの再現はせず、ほかの会社のモデル（Claude Haiku 5.5・GPT-6 Luna）との比較は、条件をそろえきれない参考の値として載せる。

測ったのは 2026-10-13。mini は標準、nano は軽量のモデルである。

## 分かったこと

| 問い | 答え | 詳しく |
|---|---|---|
| 計算や論理の問題は、軽い nano で足りるか | 足りる。数学は nano 1.000 に対し mini 0.667、論理は 0.867 に対し 0.767 で、どちらも nano が上だった。mini は途中の計算で数を取り違えた | [docs/chat.md](docs/chat.md#数学) |
| 「20 文字以内で」のような指示を守るか | 形式の制約を守れない。字数・使わない文字・表記の指定を、どちらのモデルも守りきれなかった（mini 0.800、nano 0.733） | [docs/chat.md](docs/chat.md#指示追従) |
| コードは書けるか | 書ける。10 問中 8〜9 問で、3 回とも全部のテストに通った（2026-10-13 は両モデル 0.933）。2026-10-31 は nano 0.967、3.2 mini thinking は 0.967〜1.000 | [docs/chat.md](docs/chat.md#コード) |
| 日本語の説明や言い換えは妥当か | 10 問中、mini は 8 問、nano は 6 問が妥当（下書きの判定）。要約の字数の上限は、両モデルともほとんど守れなかった | [docs/chat.md](docs/chat.md#日本語) |
| 知識はいつまでのものか | mini は 2025 年前半ごろ、nano は 2024 年前半〜後半ごろと推定した。自己申告の「2025 年 9 月」は当てにならない。知らない出来事を、作り話で答えることがある（下書きの判定） | [docs/chat.md](docs/chat.md#知識の打ち切り) |
| 安全判定・埋め込み・並べ替え・翻訳は使えるか、Chat で代用できるか | 使える。埋め込みと並べ替えは全問正解、安全判定は 18 件中 17 件で合った。安全判定は汎用の Chat のモデルのほうが正確だったが、専用モデルは 100 ms 前後で返る | [docs/non-chat.md](docs/non-chat.md) |
| Tool Calling は使えるか、何に気をつけるか | 使える。2026-10-31 には `tool_choice` を省いても 8 つの場面をすべてこなした。2026-10-13 は省くとツールを呼び続けて答えなかったので、古い版を使う所では `auto` を明示すると確実 | [docs/tool-calling.md](docs/tool-calling.md) |
| 2026-10-29 に出た 3.2 mini thinking はどうか | 数学と指示追従は推論の強さによらず 1.000、コードも 0.967〜1.000 で、mini・nano を上回った。論理は 0.733〜0.833 で nano とほぼ同じだった。推論の強さを変えたときの差は、論理で答え 3 個分、応答の時間の中央値で 0.6 秒（どれも 8 秒前後）にとどまった | [docs/chat.md](docs/chat.md#32-mini-thinking) |
| Claude Haiku 5.5・GPT-6 Luna と比べてどうか | どちらも Sarashina より一段上と読める。Haiku は自動で採点した 40 問をすべて正解した。差の大きさは、この問題集では測れない | [docs/comparison.md](docs/comparison.md) |
| 応答の速さ・時間切れ・レート制限はどうか | nano は中央値 255 ms で、mini の半分ほど。mini は同じ日に 4 回、60 秒の時間切れを起こした。429 は 171 回の送信で 0 回（ここまで 2026-10-13）。2026-10-31 の 3.2 mini thinking は 1 リクエスト 8 秒前後かかり、知識の 1 問はどの推論の強さでも 60 秒以内に返らなかった。429 は 738 回の送信で 0 回 | [docs/speed-and-limits.md](docs/speed-and-limits.md) |

## 分野ごとの得点

表の値は答えごとの正解率（1 問に 3 回答えさせ、答え 1 つずつを採点した割合）。ほかの会社のモデルは条件が違う参考の値である。

<!-- results:start -->
| 分野 | mini（2026-10-13） | nano（2026-10-13） | mini（2026-10-31） | nano（2026-10-31） | 3.2 mini thinking（指定なし） | 3.2 mini thinking（low） | 3.2 mini thinking（medium） | 3.2 mini thinking（high） | Claude Haiku 5.5 low（参考） | GPT-6 Luna low（参考） |
|---|---|---|---|---|---|---|---|---|---|---|
| 数学 | 0.667 | 1.000 | 0.633 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 |
| 論理 | 0.767 | 0.867 | 0.733 | 0.800 | 0.800 | 0.733 | 0.833 | 0.767 | 1.000 | 0.800 |
| 指示追従 | 0.800 | 0.733 | 0.833 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| コード | 0.933 | 0.933 | 0.933 | 0.967 | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 日本語 | 下書きの判定 | 下書きの判定 | 未採点 | 未採点 | 未採点 | 未採点 | 未採点 | 未採点 | 下書きの判定 | 下書きの判定 |
| 知識 | 下書きの判定 | 下書きの判定 | 未採点 | 未採点 | 未採点 | 未採点 | 未採点 | 未採点 | 下書きの判定 | 下書きの判定 |
<!-- results:end -->

日本語と知識は、Claude が下書きし、人がまだ確かめていない判定なので、数値は出していない。測った日付と条件は [docs/method.md](docs/method.md) にある。

## 自分で流すには

標準ライブラリだけで動き、Python 3.13 で試験が通る。コードの分野の採点には Docker（`python:3` のイメージ）を使う。

1. API キーを環境変数 `SARASHINA_API_KEY` か、リポジトリの直下の `.env` に置く（`.env` は Git に入れない）
2. まず送らずに、リクエストの数と所要時間の見積もりを見る

   ```bash
   python3 -m sarashina_eval.run questions --dry-run
   ```

3. 送って採点する。採点の記録は `logs/` に、得点の CSV は `results/` に出る

   ```bash
   python3 -m sarashina_eval.run questions
   python3 -m sarashina_eval.score --run-id <実行 ID>
   ```

Chat 以外のモデルは `python3 -m sarashina_eval.non_chat run` と `score`、Tool Calling は `python3 -m sarashina_eval.tools run` と `score` で同じように実行する。どれも `--help` で引数が分かる。試験は `python3 -m unittest discover -s tests` で実行する。

README の得点表は、`results/readme-runs.json` に載せる実行を書いてから `python3 -m sarashina_eval.readme_table` で書き込む。

## 文書の一覧

| 文書 | 何が書いてあるか |
|---|---|
| [docs/chat.md](docs/chat.md) | Chat の 6 分野の結果と、知識の打ち切り |
| [docs/non-chat.md](docs/non-chat.md) | Chat 以外の 4 モデルの結果と、汎用の Chat のモデルでの代用 |
| [docs/tool-calling.md](docs/tool-calling.md) | Tool Calling の振る舞いと、`tool_choice` の注意 |
| [docs/comparison.md](docs/comparison.md) | Claude Haiku 5.5・GPT-6 Luna との比較と、条件の違い |
| [docs/speed-and-limits.md](docs/speed-and-limits.md) | 応答の速さと長さ、時間切れ、レート制限 |
| [docs/method.md](docs/method.md) | 問題集・採点の方式・実行の条件・実行 ID・実行の記録・問題の一覧 |
| [docs/history.md](docs/history.md) | 本番の前の試しの実行で見えた傾向（訂正つき） |
| [GLOSSARY.md](GLOSSARY.md) | この検証で使う言葉の定義 |

## ライセンスと出力の扱い

| 対象 | ライセンス |
|---|---|
| コード（`sarashina_eval/`・`scripts/`・`tests/`） | MIT（[LICENSE](LICENSE)） |
| 文書（README・`docs/`・`GLOSSARY.md`）と、問題集・入力（`questions/`・`inputs/`・`tool_scenarios/`）、得点の CSV（`results/`） | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ja) |

ただし、次のものはどちらのライセンスの対象外である。

- 文書に引いたモデルの答え（「B」「HELLO」「690」や訳文など）。Sarashina・Claude・GPT-6 Luna それぞれの出力である
- Tool Calling でツールが返した内容（架空のお店・記事・天気）。この検証のために作った架空のものである

安全判定（guard）の有害な側の入力の文面は公開しない。
