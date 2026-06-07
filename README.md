# ML Agent Orchestration Pipeline

複数のClaudeエージェントが連携して、MLモデルを自動生成・評価・改善するパイプラインです。
scikit-learn の組み込みデータセットはもちろん、CSV形式のコンペデータにも対応しています。

## 概要

5つのエージェントが以下の流れで動作します：

```
Planner（対話でプラン決定）
    │
    └── Orchestrator（指揮）
            │
            ├── Builder  → MLコードを自動生成（sklearn / LightGBM / CatBoost）
            ├── Evaluator → コードを実行してスコアを測定（AIを使わない）
            └── Critic   → 結果を分析して改善提案を出す
                 └── Builder（改善版を再生成） → Evaluator → ...（繰り返し）
```

目標スコアに達するか、最大イテレーション数に到達すると終了します。

## 各エージェントの役割

| エージェント | 役割 |
|---|---|
| **Planner** | 実行前にユーザーと対話し、問題・データ・メトリクス・目標スコアを決定する |
| **Orchestrator** | 全体を指揮。次にどのエージェントを呼ぶか決定する |
| **Builder** | MLコードを生成。sklearn / LightGBM / CatBoost に対応。Criticの批評を元に改善版も生成 |
| **Evaluator** | 生成されたコードを実際に実行し、スコア（accuracy / AUC など）を取得 |
| **Critic** | コードとスコアを分析し、具体的な改善提案を返す |

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

### プランニングモード（推奨）

引数なしで起動するとPlannerとの対話が始まり、何を作るか一緒に決めてからパイプラインが自動実行されます。

```bash
python main.py
```

Planner はデータソース（sklearn / CSV）、メトリクス（accuracy / AUC など）、目標スコア、イテレーション数を順に確認します。

### 直接実行モード（sklearn）

```bash
python main.py --problem "Classify wine quality" --max-iterations 3 --target-score 0.95
```

### 直接実行モード（コンペ CSV）

```bash
python main.py \
  --problem "NFLドラフト予測（二値分類、Drafted列を予測）" \
  --data-dir data/nfl \
  --target-col Drafted \
  --metric auc \
  --target-score 0.85 \
  --max-iterations 5 \
  --submission-path data/nfl/submission.csv
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--problem` | なし（省略でプランニングモード） | 解かせるMLの問題 |
| `--metric` | `accuracy` | 最適化するメトリクス（accuracy / auc / rmse …） |
| `--max-iterations` | 5 | 最大繰り返し回数 |
| `--target-score` | 0.97 / 0.85 | 早期終了する目標スコア（AUCなら自動で0.85） |
| `--data-dir` | なし | train.csv / test.csv があるディレクトリ（CSVモード） |
| `--target-col` | なし | CSVの目的変数列名 |
| `--submission-path` | なし | 予測結果を保存するCSVパス |

## コンペモードの使い方（NFL例）

1. `data/nfl/` フォルダに `train.csv`、`test.csv` を置く
2. 上記の直接実行コマンドを実行
3. Builder が LightGBM コードを生成 → Evaluator が CV AUC を計測 → Critic が改善提案 → 繰り返し
4. 完了後 `data/nfl/submission.csv` に提出ファイルが保存される

> **Note:** `data/` ディレクトリは `.gitignore` に追加済みなので、データファイルはGitに含まれません。

## ファイル構成

```
ml-orchestration/
├── main.py               # エントリーポイント
├── requirements.txt      # 依存パッケージ
├── .env                  # APIキー（Gitにコミットしない）
├── data/                 # コンペ用データ置き場（Gitに含まない）
│   └── nfl/              # 例: train.csv, test.csv
├── agents/
│   ├── planner.py        # 対話でプランを決定するエージェント
│   ├── orchestrator.py   # 全体を指揮するエージェント
│   ├── builder.py        # コード生成エージェント（sklearn/LightGBM対応）
│   ├── evaluator.py      # コード実行・評価エージェント
│   └── critic.py         # 批評エージェント
└── core/
    ├── client.py         # Anthropic APIクライアント
    ├── cost_tracker.py   # トークン数・コスト集計
    └── types.py          # 共通データ型定義
```

## 使用モデル

`claude-sonnet-4-6`（全エージェント共通）
