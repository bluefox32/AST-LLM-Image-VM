# AST-LLM Image VM

AST-LLM を制御層とする画像生成VMの統合構成です。GitHub（bluefox32）の既存リポジトリ群の部品を組み合わせ、**生成 → 解析（RGB / FFT / ブロックベクトル）→ スコア → 調整・再実行 → 記録（ハッシュチェーン）** のサイクルを回す設計になっています。

現在の実装は簡易版で、画像エンジンはダミー（シード決定的な合成画像）です。エンジンはアダプタ式のため、ローカルSD（diffusers）への差し替え先が用意されています。

## 構成

| 層 | 実装 | 出所 |
|---|---|---|
| 制御 | プロンプトAST化、不変条件チェック、来歴ハッシュ | `AST-LLM`（`app/vendor/` に無改変で収録） |
| VM実行 | 命令列の実行、反復ループ、上限管理 | 新規実装 `app/vm.py` |
| エンジン | ダミーエンジン＋プール（多重起動・負荷分散） | 新規実装 `app/engines.py`, `app/engine_pool.py` |
| 解析 | RGB統計、FFT高周波比、8x8ブロック特徴 | RGB系・FFT系・Sigma-vector の考え方を簡易実装 |
| 評価 | 仮スコア（色・精細・バランス）＋ pass/retry 判定 | 評価係数の考え方＋ VMA judge の判定形式 |
| 保存 | SQLite＋ハッシュチェーン、スキーマ版管理 | SQLite-Multiple-DB / Multi-layer-hash-chain の方式を簡易化 |
| API / UI | Flask API＋単一HTML | VMA-protocol のAPI型、SNS_Demo / Benchmark のUI型 |

## ドキュメント

- 統合設計書: [`docs/integrated_design.md`](docs/integrated_design.md)
- アプリの詳細と起動方法: [`app/README.md`](app/README.md)

## クイックスタート

```bash
cd app
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py
# → http://127.0.0.1:5000

# エンジンプールで起動（負荷分散）
IMAGE_VM_WORKERS=3 .venv/bin/python server.py

# プールのベンチマーク
.venv/bin/python benchmark_pool.py --jobs 6 --delay 1.0 --workers 3
```

## 実測値（簡易版、2026-10-29）

- 生成サイクル全フロー実働（pass / 再試行打ち切り / 不変条件違反の拒否 / チェーン検証OK）
- エンジンプール: 6ジョブ×模擬負荷1.0s で 1ワーカー 7.50s → 3ワーカー 2.86s（2.62倍）

## 注意

- スコアは人間評価で較正していない**仮スコア**です。測定可能な量（色の近さ、周波数成分、コントラスト）からのみ計算しています。
- DBスキーマは版管理しています（現在 v1）。変更履歴はアプリの `/api/schema/versions` で確認できます。

## ライセンス

MIT License - see [LICENSE](LICENSE). `app/vendor/` 内の AST-LLM 本体は同リポジトリのライセンス（`app/vendor/LICENSE_AST-LLM`）に従います。
