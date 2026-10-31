# gpt6luna-low-8b0ad329 — GPT-6 Luna（low）との比較（2026-10-13）

- 目的: Sarashina と同じ 61 問を GPT-6 Luna に解かせ、同じ採点器で比べる（依頼者の求め。effort は low と指定）
- 方法: Codex CLI 0.161.0 の `codex exec -m gpt-6-luna -c model_reasoning_effort="low" -s read-only --ephemeral`。設問だけの JSON を置いた空のフォルダで動かした。コマンドは `cat prompts.json` の 1 回だけ（logs/codex/gpt-6-luna-low.log で確認）。09:45:45〜09:46:36 UTC、30,051 トークン
- 結果: 自動 4 分野 37/40（logic-003 木曜日、logic-008 72、math-009 4 を外す）。日本語は仮の判定で 9/10。知識は 2025 年 1 月まで正答、2025 年 10 月以降は「分からない」、自己申告も「分からない」。でっち上げ 0 件
- レポート: reports/2026-10-13-model-comparison.md
