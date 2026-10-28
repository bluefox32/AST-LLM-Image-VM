# AST-LLM Image VM（簡易版）

統合設計書 v0.1（`../integrated_design.md`）のフェーズ0〜1相当を、GitHub（bluefox32）の既存部品を中心に実装した簡易アプリです。

**生成 → 解析（RGB / FFT / ブロックベクトル）→ スコア → 調整・再実行 → 記録（ハッシュチェーン）** のVMサイクルが、ダミーエンジンで全て実働します。

## 起動方法

```bash
cd app
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py
# → http://127.0.0.1:5000
```

## 構成と部品の出所

| ファイル | 役割 | 出所・対応する設計 |
|---|---|---|
| `vendor/ast_llm_unified.py` | 制御層（AST、変換パイプライン、ハッシュ） | GitHub `AST-LLM` から**無改変**で取り込み（commit 5a09824）。ライセンスは `vendor/LICENSE_AST-LLM` |
| `prompt_compiler.py` | PARSE / VALIDATE / COMPILE。色語→目標色の抽出 | AST-LLM の ASTNode / TransformPipeline を使用。設計書 §6 |
| `vm.py` | VM実行層。反復ループと上限管理 | 設計書 §6.3 |
| `engines.py` | 画像エンジン・アダプタ（ダミー実働、SDは雛形） | 設計書 §9 |
| `analysis.py` | ANALYZE_RGB / FFT / VEC（8x8ブロック特徴） | RGB系・FFT系・Sigma-vector の考え方を NumPy+Pillow で簡易実装 |
| `scoring.py` | 仮スコア（色・精細・バランス）と pass/retry 判定 | VMA judge の形式を借用。**未較正のため仮スコア** |
| `store.py` | SQLite保存、画像SHA256、ハッシュチェーン検証 | SQLite-Multiple-DB / Multi-layer-hash-chain の方式を簡易化 |
| `server.py` | Flask API | VMA-protocol files 104 の API 型 |
| `static/index.html` | 単一HTMLのUI | SNS_Demo の単一ファイル型＋Benchmark のスコア表示型 |

## API

| メソッド | パス | 内容 |
|---|---|---|
| POST | `/api/generate` | `{prompt, size, style, seed}` で生成サイクルを実行 |
| GET | `/api/gallery` | ギャラリー一覧 |
| GET | `/api/gallery/<id>` | 詳細（プロンプトAST、解析結果、反復履歴） |
| GET | `/api/gallery/<id>/image` | 画像PNG |
| GET | `/api/gallery/stats` | 件数・平均スコア・pass数 |
| GET | `/api/chain/verify` | ハッシュチェーン検証 |
| GET | `/api/schema/versions` | DBスキーマの変更履歴（版・概要・理由・適用日時） |
| GET | `/api/health`, `/api/info` | 状態・構成情報 |

サイズは `512x512 / 768x512 / 512x768 / 1024x1024`、スタイルは `vivid / natural` のみ許可され、違反は AST-LLM の変換パイプラインの不変条件チェックで 400 になります。

## 実測済みの動作（2026-10-29）

- 「青い空と赤い屋根の家」（vivid, seed 42）→ 1回で pass、目標色（青・赤）を反映したグラデーション画像を生成
- 「静かな森の朝」（natural, seed 7）→ 2回反復しても retry、上限で打ち切り最良回を保存
- 不正サイズ `999x999` → 400（不変条件違反）
- チェーン検証 OK、統計API 正常、画像PNG 正常

## エンジンプール（負荷分散の試作）

`engine_pool.py` でエンジンを複数プロセス起動し、空きワーカーへ生成を振り分けます。VM・サーバーから見たインターフェースは単体エンジンと同じで、担当ワーカー（slot / pid / 所要秒）は生成記録の runs に残ります。

```bash
# サーバーをプール x3 で起動
IMAGE_VM_WORKERS=3 .venv/bin/python server.py

# ベンチマーク（模擬負荷つきで分散を計測）
.venv/bin/python benchmark_pool.py --jobs 6 --delay 1.0 --workers 3
```

実測（2026-10-29、このVM）: 6ジョブ×模擬負荷1.0s で、ワーカー1台 7.50s → 3台 2.86s（2.62倍、配分は2件ずつ均等）。サーバー経由の同時3リクエストも3ワーカーへ分散され、全てpass、チェーン検証OKでした。

分散の単位は「リクエスト（ジョブ）」です。高解像度の1枚を複数エンジンで分担するには、タイル分割＋拡大の工程分割が別途必要です（設計書 §9 の想定通り）。

## DBスキーマの版管理（重点管理項目）

「DB構造に傾向が反映される」可能性を後から検証できるよう、スキーマ変更は必ず履歴に残します。

- 定義は `store.py` の `SCHEMA_HISTORY`（版・概要・理由）。変更時は `CURRENT_SCHEMA_VERSION` を上げて追記する運用です
- 適用履歴はDB内の `schema_versions` テーブルに自動記録され、`/api/schema/versions` で確認できます
- 現在は v1（generations 単一テーブル＋ハッシュチェーン）。次に予定する正規化（6テーブル分離）は v2 として理由とともに記録されます

## 既知の制約

- ダミーエンジンの画像はグラデーション＋図形の合成です。実画像生成には `engines.py` の `SDEngineStub` を実装し、diffusers＋モデル重みとGPU環境が必要です。
- スコアは人間評価で較正していない仮スコアです。色スコアは「プロンプト中の色語との近さ」等、測定可能な量のみから計算しています。
- ADJUST（再試行時の調整）は常に改善するとは限りません。VMは最良回を採用するため悪化はしませんが、調整則自体は今後の課題です。
- データは `data/`（SQLite と画像）へ保存されます。消せば初期化できます。
