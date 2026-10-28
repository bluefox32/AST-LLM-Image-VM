"""評価層 (設計書 §7)。

測定可能な指標だけから仮スコアを作る。人間評価による較正 (フェーズ4) が
済むまでは「仮スコア」であり、表示側でもその旨を明示する前提。

判定は VMA judge の形式 (連続スコア + フィデリティ + 閾値判定) を借りる。
"""
from __future__ import annotations

import math

PASS_THRESHOLD = 0.60

# スタイル別の目標彩度 (色語が無い場合の色スコア基準)
STYLE_SATURATION_TARGET = {"vivid": 0.45, "natural": 0.22}


def _color_distance(a: list[int], b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def score(analysis: dict, target_colors: list, style: str) -> dict:
    dominant = analysis.get("dominant_colors", [])

    # --- 色スコア ---
    if target_colors and dominant:
        # 各目標色について、最も近い主要色との距離を見る (最大距離 441.7 で正規化)
        closeness = []
        for target in target_colors:
            best = min(_color_distance(d["rgb"], target) for d in dominant)
            closeness.append(1.0 - min(best / 441.7, 1.0))
        color_score = sum(closeness) / len(closeness)
        color_basis = "目標色との距離"
    else:
        sat = analysis.get("saturation", 0.0)
        target_sat = STYLE_SATURATION_TARGET.get(style, 0.3)
        color_score = max(0.0, 1.0 - abs(sat - target_sat) / max(target_sat, 0.1))
        color_basis = f"スタイル目標彩度 ({style}: {target_sat}) との差"

    # --- 精細スコア: 高周波エネルギー比を 0.30 で頭打ちに正規化 ---
    sharp_score = max(0.0, min(analysis.get("hf_ratio", 0.0) / 0.30, 1.0))

    # --- バランススコア: 明暗コントラストが中庸 (std 45前後) で高い ---
    contrast = analysis.get("contrast", 0.0)
    balance_score = max(0.0, 1.0 - abs(contrast - 45.0) / 45.0)

    total = 0.5 * color_score + 0.3 * sharp_score + 0.2 * balance_score

    # フィデリティ: 測定の健全性 (必須指標が揃い、有限値であること)
    required = ["rgb_mean", "saturation", "hf_ratio", "contrast", "dominant_colors"]
    fidelity = 1.0 if all(k in analysis and analysis[k] not in (None, []) for k in required) else 0.5

    return {
        "provisional": True,
        "color_score": round(color_score, 4),
        "color_basis": color_basis,
        "sharp_score": round(sharp_score, 4),
        "balance_score": round(balance_score, 4),
        "total": round(total, 4),
        "fidelity": fidelity,
        "threshold": PASS_THRESHOLD,
        "verdict": "pass" if total >= PASS_THRESHOLD else "retry",
    }
