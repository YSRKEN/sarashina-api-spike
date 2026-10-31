# sim — Haiku 5.5・GPT-6 Luna による Chat 以外のモデルの真似（2026-10-13）

- 設問 logs/sim/prompts.json（38 問。guard 18 は非公開の 12 を含む）、鍵 logs/sim/key.json、答え logs/sim/haiku55-low.json・gpt6luna-low.json、採点 logs/sim/scores.json
- guard 判定: Sarashina 17/18、Haiku 18/18、Luna 18/18。カテゴリ完全一致: 12・12・13
- embedding・rerank: 3 者とも 5/5
- translate: 3 者とも意味は通る。皮肉は Haiku だけ落とし、慣用句は Luna だけ意味で訳した
- レポート: reports/2026-10-13-model-comparison.md
