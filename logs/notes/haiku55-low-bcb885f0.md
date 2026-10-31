# haiku55-low-bcb885f0 — Claude Haiku 5.5（low）との比較（2026-10-13）

- 目的: Sarashina と同じ 61 問を Claude Haiku 5.5 に解かせ、同じ採点器で比べる（依頼者の求め）
- 方法: Claude Code の担当（model=haiku, effort=low）に、正解を含まない設問だけの JSON を 1 回読ませ、道具なしで答えさせた。API ではない。記録は logs/2026-10-13-haiku.jsonl（送受信のヘッダーや所要時間は無い）
- 結果: 自動 4 分野 40/40。日本語は仮の判定で 9/10（japanese-003 だけ一部）。知識は 2025 年 10 月まで正答、2026 年は 3 問とも「分かりません」、自己申告「2026 年 6 月ごろ」
- 注意: 問題集の多くは Claude（Opus 5.5）の下書きで、同じ系列に有利な可能性。1 問 1 回、まとめて解かせたので、Sarashina（n=3、1 問ずつ）と条件が違う
- レポート: reports/2026-10-13-model-comparison.md
