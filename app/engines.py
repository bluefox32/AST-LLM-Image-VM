"""画像エンジン・アダプタ層 (設計書 §9)。

アダプタIF: generate(params) -> (PIL.Image, meta dict)
params は prompt_compiler.compile_prompt の出力に、反復調整用の
palette_boost (0.0-1.0) が加わったもの。

簡易版では DummyEngine のみ実働。SD エンジンは差し替え先の雛形。
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


class DummyEngine:
    """ダミーエンジン。

    シードとプロンプトから決定的に、グラデーション＋柔らかい円＋粒子ノイズの
    画像を合成する。実モデルではないが、色分布・周波数・ブロック特徴といった
    解析対象としては本物と同じ性質 (測定可能なRGB配列) を持つ。
    """

    name = "dummy"

    def generate(self, params: dict) -> tuple[Image.Image, dict]:
        w, h = (int(v) for v in params["size"].split("x"))
        seed = int(params["seed"])
        rng = np.random.default_rng(seed)

        palette = self._palette(params, rng)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xn, yn = xx / max(w - 1, 1), yy / max(h - 1, 1)

        # 背景: パレット2色の対角グラデーション
        t = (xn * 0.65 + yn * 0.35)[..., None]
        img = palette[0][None, None, :] * (1 - t) + palette[1][None, None, :] * t

        # 柔らかい円をランダムに配置 (ガウス減衰)
        n_circles = int(rng.integers(5, 10))
        for i in range(n_circles):
            cx, cy = rng.random(), rng.random()
            radius = float(rng.uniform(0.08, 0.30))
            color = palette[int(rng.integers(0, len(palette)))]
            d2 = (xn - cx) ** 2 + (yn - cy) ** 2
            alpha = np.exp(-d2 / (2 * radius**2))[..., None] * float(rng.uniform(0.35, 0.8))
            img = img * (1 - alpha) + color[None, None, :] * alpha

        # 粒子ノイズ (高周波成分を持たせる)
        noise = rng.normal(0, 6.0, size=(h, w, 1)).astype(np.float32)
        img = img + noise

        # ビネット
        vig = 1.0 - 0.18 * (((xn - 0.5) ** 2 + (yn - 0.5) ** 2) * 2)[..., None]
        img = img * vig

        arr = np.clip(img, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        meta = {
            "engine": self.name,
            "model": "dummy-gradient-shapes-v1",
            "seed_used": seed,
            "palette_boost": params.get("palette_boost", 0.0),
            "seconds": 0.0,
            "cost": 0.0,
        }
        return image, meta

    def _palette(self, params: dict, rng: np.random.Generator) -> np.ndarray:
        """目標色があればそれを、無ければプロンプトのハッシュから色相を作る。

        palette_boost が上がるほど目標色への寄せを強くする (ADJUST 命令用)。
        """
        targets = [np.array(c, dtype=np.float32) for c in params.get("target_colors", [])]
        if not targets:
            digest = hashlib.sha256(params["prompt"].encode()).digest()
            hue = digest[0] / 255.0
            base = np.array(self._hsv_to_rgb(hue, 0.55, 0.75), dtype=np.float32) * 255
            targets = [base, base * 0.55 + np.array([20, 20, 30], dtype=np.float32)]
        while len(targets) < 3:
            jitter = rng.normal(0, 18, size=3).astype(np.float32)
            targets.append(np.clip(targets[-1] * 0.8 + jitter + 30, 0, 255))

        palette = np.stack(targets[:4])
        boost = float(params.get("palette_boost", 0.0))
        if boost > 0 and params.get("target_colors"):
            # 目標色そのものへ寄せる: ばらつきを boost 分だけ縮める
            center = np.stack([np.array(c, dtype=np.float32) for c in params["target_colors"]]).mean(axis=0)
            palette = palette * (1 - boost * 0.5) + center[None, :] * (boost * 0.5)

        if params["style"] == "natural":
            gray = palette.mean(axis=1, keepdims=True)
            palette = palette * 0.55 + gray * 0.45  # 彩度を落とす
        return palette.astype(np.float32)

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
        import colorsys

        return colorsys.hsv_to_rgb(h, s, v)


class SDEngineStub:
    """ローカルSD (diffusers) 用の差し替え雛形。簡易版では未接続。"""

    name = "sd-local"

    def generate(self, params: dict) -> tuple[Image.Image, dict]:
        raise RuntimeError(
            "SDエンジンは簡易版では未接続です。diffusers とモデル重みを用意し、"
            "このクラスの generate を実装して server.py の ENGINE を差し替えてください。"
        )
