# mesh-gen — メッシュ→生成 パイロット (GitHub Actions 並列版)

3D メッシュ (頭部プロキシ) が構造を担い、生成 (SD1.5 + ControlNet-Depth) が
絵を載せる方式の実証。Colab の無料 GPU 枠が使えない場合の CPU 並列ルート。

## 実行方法

1. このリポジトリの **Actions** タブ → **mesh-gen-parallel** → **Run workflow**
2. 8 ジョブ (45° 刻み) が並列に走り、各ジョブが 1 フレームを生成する
3. 完了後、各ジョブの成果物 `frame-0` … `frame-7` (PNG + info.json) をダウンロード

## 中身

- `perspective_fixed.py` — 検証済みの透視投影 (約定: +z 前方、範囲外は fail-closed)
- `make_guide.py` — 頭部プロキシメッシュ (頭+両耳団子+鼻、3,072 頂点) から
  深度ガイドを角度ごとに生成。Colab 版ノートブックのガイドと同一構成
- `generate_one.py` — ガイド 1 枚 → 生成 1 フレーム (512×512、20 ステップ、seed 1234 固定)

ガイド画像はリポジトリに置かず、コードから毎回生成する (再現性の担保)。
依存関係は diffusers 0.32.2 + transformers 4.48.3 の対固定
(片方だけ固定すると import で衝突するため)。
