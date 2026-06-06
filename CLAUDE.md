# CLAUDE.md

## プロジェクト概要

複数のClaudeエージェントが連携してMLモデルを自動生成・評価・改善するパイプライン。
非技術者（製造業・建設業などレガシー産業）への納品を想定して開発中。

## エージェント構成

- **Planner** (`agents/planner.py`) — 実行前にユーザーと対話してプランを決定
- **Orchestrator** (`agents/orchestrator.py`) — 全体を指揮し、各エージェントの呼び出しを制御
- **Builder** (`agents/builder.py`) — scikit-learnのMLコードを生成・改善
- **Evaluator** (`agents/evaluator.py`) — コードを実行して精度を測定（AIを使わない）
- **Critic** (`agents/critic.py`) — 結果を分析して改善提案を出す

使用モデル：`claude-sonnet-4-6`（`core/client.py` で定義）

## 作業ルール

- **コードを変更したら必ずREADMEも更新してpushする**
- `.env` はGitにコミットしない（`.gitignore` 済み）

## 今後の方針

- 非技術者向けWebアプリ化を予定（まずStreamlitでMVP）
- 顧客に見せるUIでは内部の技術的出力（トークン数・エージェント名など）を隠す
- 製造業・建設業のデータ（CSVなど）を扱えるようにする
