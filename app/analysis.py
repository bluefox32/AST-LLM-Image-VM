"""解析層 (設計書 §6.2 の ANALYZE_RGB / ANALYZE_FFT / ANALYZE_VEC)。

生成画像を測定可能な量に変換する。NumPy + Pillow のみで実装し、
RGB-compress-algorithms / furie 系 / Sigma-vector の考え方を簡易化して取り込む。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

GRID = 8  # Σベクトル用のブロック分割 (8x8)


def analyze(image: Image.Image) -> dict:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    h, w, _ = arr.shape

    # --- RGB統計 ---
    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))
    luminance = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    brightness = float(luminance.mean())
    contrast = float(luminance.std())

    # 彩度 (簡易): (max-min)/max の平均
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saturation = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())

    # --- 主要色 (量子化して上位5色) ---
    quantized = image.convert("RGB").quantize(colors=5, method=Image.MEDIANCUT)
    palette = quantized.getpalette() or []
    counts = sorted(quantized.getcolors() or [], reverse=True)
    dominant = []
    for count, idx in counts[:5]:
        rgb = palette[idx * 3: idx * 3 + 3]
        if len(rgb) == 3:
            dominant.append({"rgb": [int(v) for v in rgb], "ratio": round(count / (h * w), 4)})

    # --- 周波数解析 (FFT): 高周波エネルギー比 ---
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(luminance - luminance.mean()))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = min(h, w) * 0.25
    total_energy = float(spectrum.sum()) or 1.0
    hf_ratio = float(spectrum[radius > cutoff].sum() / total_energy)

    # --- ブロック特徴ベクトル (Σベクトル簡易版): 8x8ブロックの平均RGB ---
    blocks = []
    bh, bw = h // GRID, w // GRID
    for gy in range(GRID):
        for gx in range(GRID):
            cell = arr[gy * bh:(gy + 1) * bh, gx * bw:(gx + 1) * bw]
            blocks.extend(float(v) for v in cell.mean(axis=(0, 1)))
    block_vector = [round(v, 2) for v in blocks]

    return {
        "width": w,
        "height": h,
        "rgb_mean": [round(float(v), 2) for v in rgb_mean],
        "rgb_std": [round(float(v), 2) for v in rgb_std],
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "saturation": round(saturation, 4),
        "dominant_colors": dominant,
        "hf_ratio": round(hf_ratio, 4),
        "block_vector": block_vector,
    }
