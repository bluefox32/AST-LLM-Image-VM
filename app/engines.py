"""画像エンジン・アダプタ層 (設計書 §9)。

アダプタIF: generate(params) -> (PIL.Image, meta dict)
params は prompt_compiler.compile_prompt の出力に、反復調整用の
palette_boost (0.0-1.0) が加わったもの。

簡易版では DummyEngine のみ実働。SD エンジンは差し替え先の雛形。
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
from PIL import Image


def _db_lookup(db, prompt: str):
    """エンジン側 DB (engine_db) の参照と meta 断片。db=None なら空。"""
    if db is None:
        return None, {}
    import engine_db

    ctx = engine_db.db_context(db, prompt)
    meta = {
        "db_hit": ctx["status"] in ("hit", "recovered"),
        "db_status": "corrupt_fallback" if ctx["status"] == "corrupt" else ctx["status"],
        "db_version": ctx["version"],
        "db_key": engine_db.key_for_prompt(prompt)[:12],
    }
    return ctx, meta


class DummyEngine:
    """ダミーエンジン。

    シードとプロンプトから決定的に、グラデーション＋柔らかい円＋粒子ノイズの
    画像を合成する。実モデルではないが、色分布・周波数・ブロック特徴といった
    解析対象としては本物と同じ性質 (測定可能なRGB配列) を持つ。

    db (engine_db.EngineDB) を渡すと、プロンプトのキーで登録済みパレット
    (kind="palette") を引き、当たれば目標色をその登録内容で置き換える。
    外れ・破損時は従来動作に落ち、meta の db_status に残る。db=None の
    ときは従来と完全に同一動作 (meta にも何も足さない)。
    """

    name = "dummy"

    def __init__(self, db=None):
        self.db = db

    def generate(self, params: dict) -> tuple[Image.Image, dict]:
        w, h = (int(v) for v in params["size"].split("x"))
        seed = int(params["seed"])
        rng = np.random.default_rng(seed)

        ctx, db_meta = _db_lookup(self.db, params["prompt"])
        eff = params
        if (
            ctx is not None
            and ctx["status"] in ("hit", "recovered")
            and ctx["kind"] == "palette"
            and ctx["payload"]
        ):
            eff = dict(params)
            eff["target_colors"] = [list(c) for c in ctx["payload"].get("target_colors", [])]

        palette = self._palette(eff, rng)
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
            **db_meta,
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


class SDEngine:
    """ローカルSD (diffusers) エンジン。SDEngineStub が示した差し替え席の実装。

    重い依存 (torch / diffusers) とモデル重みは初回 generate で遅延ロード
    するので、DummyEngine だけの軽量運用には影響しない。生成条件
    (チェックポイント / steps / guidance / ネガティブ) はエンジン側の
    構築時設定であり、params (プロンプトコンパイラの出力) には混ぜない。
    チェックポイントの取得は mesh-gen の generate_one.py と同じ方法
    (Civitai の版別ダウンロード、匿名で可) で、既定は AbyssOrangeMix2 SFW
    (Civitai model 4437 / version 5021)。
    """

    name = "sd-local"

    DEFAULT_CKPT_URL = "https://civitai.com/api/download/models/5021"
    DEFAULT_MODEL_LABEL = "abyssorangemix2-sfw (civitai 4437@5021)"
    DEFAULT_NEGATIVE = (
        "blurry, deformed, extra limbs, text, watermark, realistic, photo"
    )
    _UA = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        )
    }

    def __init__(
        self,
        ckpt_path: str | None = None,
        ckpt_url: str | None = None,
        steps: int = 20,
        guidance: float = 7.0,
        negative: str = DEFAULT_NEGATIVE,
        model_label: str = DEFAULT_MODEL_LABEL,
        db=None,
    ):
        self.ckpt_path = ckpt_path or os.environ.get("IMAGE_VM_SD_CKPT") or os.path.join(
            os.path.expanduser("~"), ".cache", "image-vm", "aom2_sfw_fp16.safetensors"
        )
        self.ckpt_url = ckpt_url or self.DEFAULT_CKPT_URL
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.negative = negative
        self.model_label = model_label
        self.db = db  # engine_db.EngineDB (v1 では参照と meta 記録のみ)
        self._pipe = None

    def _ensure_ckpt(self) -> str:
        if os.path.exists(self.ckpt_path) and os.path.getsize(self.ckpt_path) > 1_000_000_000:
            return self.ckpt_path
        import urllib.request

        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
        req = urllib.request.Request(self.ckpt_url, headers=self._UA)
        with urllib.request.build_opener().open(req, timeout=120) as r, open(
            self.ckpt_path, "wb"
        ) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return self.ckpt_path

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

        ckpt = self._ensure_ckpt()
        pipe = StableDiffusionPipeline.from_single_file(
            ckpt,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self._pipe = pipe
        return pipe

    def generate(self, params: dict) -> tuple[Image.Image, dict]:
        import torch

        w, h = (int(v) for v in params["size"].split("x"))
        seed = int(params["seed"])
        pipe = self._load()
        started = time.perf_counter()
        image = pipe(
            params["prompt"],
            negative_prompt=self.negative,
            width=w,
            height=h,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        _, db_meta = _db_lookup(self.db, params["prompt"])
        meta = {
            "engine": self.name,
            "model": self.model_label,
            "seed_used": seed,
            "palette_boost": params.get("palette_boost", 0.0),
            "seconds": round(time.perf_counter() - started, 3),
            "cost": 0.0,
            **db_meta,
        }
        return image, meta
