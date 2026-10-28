"""プロンプト・コンパイラ層。

AST-LLM (vendor/ast_llm_unified.py) の ASTNode / Scope / TransformPipeline を
そのまま使い、画像生成用プロンプトを AST 化し、不変条件を検証して
エンジン用パラメータセット (JSON) へコンパイルする。

設計書 §6.2 の命令 PARSE / VALIDATE / COMPILE に対応する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor"))

from ast_llm_unified import ASTNode, Scope, TransformPipeline, TransformPass  # noqa: E402

ALLOWED_SIZES = {"512x512", "768x512", "512x768", "1024x1024"}
ALLOWED_STYLES = {"vivid", "natural"}

# プロンプト内の色語 → 目標RGB (create-standard-colors- の「基準」側の考え方を
# 簡易版として固定テーブルで実装)
COLOR_WORDS: dict[str, tuple[int, int, int]] = {
    "青": (40, 90, 200), "蒼": (40, 90, 200), "blue": (40, 90, 200),
    "水色": (110, 190, 225),
    "赤": (200, 50, 45), "red": (200, 50, 45),
    "緑": (60, 160, 80), "green": (60, 160, 80),
    "黄": (235, 200, 60), "yellow": (235, 200, 60),
    "紫": (130, 80, 180), "purple": (130, 80, 180),
    "ピンク": (235, 140, 170), "桃": (235, 140, 170), "pink": (235, 140, 170),
    "オレンジ": (235, 140, 50), "橙": (235, 140, 50), "orange": (235, 140, 50),
    "白": (240, 240, 235), "white": (240, 240, 235),
    "黒": (30, 30, 35), "black": (30, 30, 35),
    "金": (210, 175, 90), "gold": (210, 175, 90),
    "銀": (180, 185, 190), "silver": (180, 185, 190),
    "茶": (140, 100, 65), "brown": (140, 100, 65),
}


def extract_target_colors(prompt: str) -> list[tuple[int, int, int]]:
    """プロンプト中の色語を出現順に抽出する (重複除去)。"""
    found: list[tuple[int, int, int]] = []
    lowered = prompt.lower()
    # 長い語を先に見る (例: 水色 を 青 より先に判定するため位置でソート)
    hits: list[tuple[int, tuple[int, int, int]]] = []
    for word, rgb in COLOR_WORDS.items():
        idx = lowered.find(word.lower())
        if idx >= 0:
            hits.append((idx, rgb))
    for _, rgb in sorted(hits):
        if rgb not in found:
            found.append(rgb)
    return found


def build_prompt_ast(prompt: str, size: str, style: str, seed: int) -> ASTNode:
    """PARSE: プロンプトと生成条件を AST 化する。"""
    targets = extract_target_colors(prompt)
    root = ASTNode(
        "ImagePrompt",
        value=prompt,
        children=[
            ASTNode("Subject", value=prompt),
            ASTNode("Style", value=style),
            ASTNode("Size", value=size),
            ASTNode("Seed", value=str(seed)),
            ASTNode(
                "PaletteTarget",
                value=",".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in targets),
                meta={"colors": [list(c) for c in targets]},
            ),
        ],
    )
    root._hash = root.compute_hash()
    root.node_id = root._hash
    return root


def ast_to_dict(node: ASTNode) -> dict:
    return {
        "type": node.node_type,
        "value": node.value,
        "meta": node.meta,
        "hash": node._hash,
        "children": [ast_to_dict(c) for c in node.children],
    }


def _compile_pass(node: ASTNode, scope: Scope) -> str:
    """COMPILE: AST → エンジン用パラメータセット (決定的な出力にする)。"""
    fields = {c.node_type: c for c in node.children}
    params = {
        "prompt": fields["Subject"].value,
        "style": fields["Style"].value,
        "size": fields["Size"].value,
        "seed": int(fields["Seed"].value),
        "target_colors": fields["PaletteTarget"].meta.get("colors", []),
        "ast_hash": node._hash,
    }
    scope.define("params", params)
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def _validate_params(node: ASTNode, output: str) -> list[str]:
    """VALIDATE: 不変条件チェック。違反はリストで返す。"""
    errors: list[str] = []
    params = json.loads(output)
    if not params["prompt"] or not params["prompt"].strip():
        errors.append("プロンプトが空です")
    if params["size"] not in ALLOWED_SIZES:
        errors.append(f"未許可のサイズです: {params['size']} (許可: {sorted(ALLOWED_SIZES)})")
    if params["style"] not in ALLOWED_STYLES:
        errors.append(f"未許可のスタイルです: {params['style']} (許可: {sorted(ALLOWED_STYLES)})")
    if not (0 <= params["seed"] <= 2**31 - 1):
        errors.append("シードが範囲外です")
    return errors


def compile_prompt(prompt: str, size: str, style: str, seed: int) -> tuple[ASTNode, dict]:
    """PARSE → VALIDATE → COMPILE を実行する。

    TransformPipeline が冪等性チェックと不変条件チェックを行う。
    違反時は ast_llm_unified.TransformError が送出される。
    """
    root = build_prompt_ast(prompt, size, style, seed)
    if not root.verify():
        raise ValueError("AST のハッシュ検証に失敗しました")
    scope = Scope("image_generation")
    pipeline = TransformPipeline()
    pipeline.add_pass(TransformPass(name="compile", fn=_compile_pass, validator=_validate_params))
    output = pipeline.run(root, scope)
    return root, json.loads(output)
