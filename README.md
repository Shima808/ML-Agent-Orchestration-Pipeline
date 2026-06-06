# ML Agent Orchestration Pipeline

複数のClaudeエージェントが連携して、MLモデルを自動生成・評価・改善するパイプラインです。

## 概要

4つのエージェントが以下の流れで動作します：

```
Orchestrator（指揮）
    │
    ├── Builder  → MLコードを自動生成
    ├── Evaluator → コードを実行して精度を測定
    └── Critic   → 改善点を分析・提案
         └── Builder（改善版を再生成） → Evaluator → ...（繰り返し）
```

目標精度に達するか、最大イテレーション数に到達すると終了します。

## 各エージェントの役割

| エージェント | 役割 |
|---|---|
| **Orchestrator** | 全体を指揮。次にどのエージェントを呼ぶか決定する |
| **Builder** | scikit-learnのMLコードを生成。Criticの批評を元に改善版も生成 |
| **Evaluator** | 生成されたコードを実際に実行し、精度（accuracy）を取得 |
| **Critic** | コードと精度を分析し、具体的な改善提案を返す |

## セットアップ

**1. APIキーを設定する**

`.env` ファイルに自分のAnthropicのAPIキーを記載します：

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

**2. 依存関係をインストールする**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 実行方法

```bash
python main.py
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--problem` | Irisの花の分類 | 解かせるMLの問題 |
| `--max-iterations` | 5 | 最大繰り返し回数 |
| `--target-score` | 0.97 | 早期終了する目標精度 |

**例：**

```bash
# 繰り返しを3回、目標精度95%に変更
python main.py --max-iterations 3 --target-score 0.95
```

## 出力例

```
============================================================
  ML Agent Orchestration Pipeline
============================================================
  Problem:        Classify Iris flower species with maximum accuracy.
  Max iterations: 5
  Target score:   0.97
  Model:          claude-sonnet-4-6
============================================================

[Orchestrator] Deciding next action (iter=0)...
[Orchestrator] → call_builder
  [Orchestrator] 312 in / 28 out  ($0.0010)

[Builder] Generating model code...
  Approach: Logistic Regression baseline
  [Builder] 512 in / 480 out  ($0.0023)

[Evaluator] Running code...
  Accuracy: 0.9333

[Critic] Analyzing results...
  Suggestions: Try Random Forest with n_estimators=200...

...（繰り返し）...

============================================================
  Final Results
============================================================
  Best accuracy: 0.9733
  Best model:    RandomForestClassifier
  At iteration:  3

══════════════════════════════════════════════════════════════
Total tokens: 12,450 in / 3,210 out
Total cost:   $0.0521
══════════════════════════════════════════════════════════════
```

## ファイル構成

```
ml-orchestration/
├── main.py               # エントリーポイント
├── requirements.txt      # 依存パッケージ
├── .env                  # APIキー（Gitにコミットしない）
├── agents/
│   ├── orchestrator.py   # 全体を指揮するエージェント
│   ├── builder.py        # コード生成エージェント
│   ├── evaluator.py      # コード実行・評価エージェント
│   └── critic.py         # 批評エージェント
└── core/
    ├── client.py         # Anthropic APIクライアント
    ├── cost_tracker.py   # トークン数・コスト集計
    └── types.py          # 共通データ型定義
```

## 使用モデル

`claude-sonnet-4-6`（全エージェント共通）
