"""APIサーバー (VMA-protocol files 104 の Flask 型を踏襲)。

エンドポイント:
  GET  /                          UI (static/index.html)
  POST /api/generate              生成 (VMサイクル実行)
  GET  /api/gallery               ギャラリー一覧
  GET  /api/gallery/<id>          生成の詳細 (AST・解析・履歴)
  GET  /api/gallery/<id>/image    画像PNG
  GET  /api/gallery/stats         統計
  GET  /api/chain/verify          ハッシュチェーン検証
  GET  /api/health                ヘルスチェック
  GET  /api/info                  構成情報
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor"))

from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402

import os  # noqa: E402

import store  # noqa: E402
from vm import ImageVM  # noqa: E402
from ast_llm_unified import TransformError  # noqa: E402

BASE = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")

# 環境変数 IMAGE_VM_WORKERS でエンジンプールの多重度を切り替える (既定は単体)
_pool_workers = int(os.environ.get("IMAGE_VM_WORKERS", "1"))
if _pool_workers > 1:
    from engine_pool import EnginePool

    vm = ImageVM(pool=EnginePool(workers=_pool_workers))
    ENGINE_MODE = f"pool x{_pool_workers}"
else:
    vm = ImageVM()
    ENGINE_MODE = "single"


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/generate")
def api_generate():
    body = request.get_json(force=True, silent=True) or {}
    prompt = str(body.get("prompt", "")).strip()
    size = str(body.get("size", "512x512"))
    style = str(body.get("style", "vivid"))
    try:
        seed = int(body.get("seed", 42))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "error": "seed は整数で指定してください"}), 400
    if not prompt:
        return jsonify({"status": "error", "error": "prompt は必須です"}), 400
    try:
        result = vm.run(prompt, size=size, style=style, seed=seed)
    except TransformError as exc:
        return jsonify({"status": "error", "error": f"不変条件違反: {exc}"}), 400
    except ValueError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400
    return jsonify({"status": "success", **result})


@app.get("/api/gallery")
def api_gallery():
    return jsonify({"status": "success", "items": store.list_generations()})


@app.get("/api/gallery/<int:gen_id>")
def api_gallery_detail(gen_id: int):
    item = store.get_generation(gen_id)
    if item is None:
        return jsonify({"status": "error", "error": "not found"}), 404
    return jsonify({"status": "success", **item})


@app.get("/api/gallery/<int:gen_id>/image")
def api_gallery_image(gen_id: int):
    path = store.image_path(gen_id)
    if path is None:
        return jsonify({"status": "error", "error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.get("/api/gallery/stats")
def api_gallery_stats():
    return jsonify({"status": "success", **store.stats()})


@app.get("/api/chain/verify")
def api_chain_verify():
    return jsonify({"status": "success", **store.verify_chain()})


@app.get("/api/schema/versions")
def api_schema_versions():
    return jsonify({"status": "success", "versions": store.list_schema_versions()})


@app.get("/api/health")
def api_health():
    return jsonify({"status": "ok", "engine": vm.engine.name, "engine_mode": ENGINE_MODE})


@app.get("/api/info")
def api_info():
    return jsonify({
        "status": "success",
        "app": "AST-LLM Image VM (simple edition)",
        "engine": vm.engine.name,
        "engine_mode": ENGINE_MODE,
        "components": {
            "control": "AST-LLM (vendor, unmodified)",
            "server": "Flask (VMA-protocol API pattern)",
            "analysis": "NumPy/Pillow RGB + FFT + block vectors",
            "scoring": "provisional, uncalibrated",
            "store": "SQLite + hash chain",
        },
    })


if __name__ == "__main__":
    store.init_db()
    app.run(host="127.0.0.1", port=5000, debug=False)
