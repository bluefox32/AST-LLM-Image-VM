"""VM実行層 (設計書 §6)。

1回の生成を命令列として実行する:
  PARSE → VALIDATE → COMPILE → [ GEN → ANALYZE → SCORE → (ADJUST) ] → COMMIT

反復には回数とスコア閾値の上限管理を入れる (設計書 §6.3)。
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import analysis as analysis_mod
import engines
import prompt_compiler
import scoring as scoring_mod
import store

MAX_ITERATIONS = 2  # 簡易版の反復上限 (初回 + 再試行1回)


class ImageVM:
    def __init__(self, engine=None, pool=None):
        self.engine = engine or engines.DummyEngine()
        self.pool = pool  # EnginePool があれば GEN はプール経由 (負荷分散)

    def run(self, prompt: str, size: str = "512x512", style: str = "vivid",
            seed: int = 42) -> dict:
        # PARSE / VALIDATE / COMPILE (AST-LLM の TransformPipeline が検証)
        root, params = prompt_compiler.compile_prompt(prompt, size, style, seed)

        runs = []
        best = None
        current = dict(params)
        current["palette_boost"] = 0.0

        for iteration in range(1, MAX_ITERATIONS + 1):
            # GEN (プールがあれば空きワーカーへ振り分け)
            if self.pool is not None:
                image, meta = self.pool.generate(current)
            else:
                image, meta = self.engine.generate(current)
            # ANALYZE (RGB / FFT / VEC)
            measured = analysis_mod.analyze(image)
            # SCORE
            result_score = scoring_mod.score(
                measured, current.get("target_colors", []), current["style"]
            )
            run_record = {
                "iteration": iteration,
                "seed_used": current["seed"],
                "palette_boost": current["palette_boost"],
                "score_total": result_score["total"],
                "verdict": result_score["verdict"],
            }
            if "worker_slot" in meta:  # プール経由時は担当ワーカーを記録 (再現性)
                run_record["worker_slot"] = meta["worker_slot"]
                run_record["worker_pid"] = meta["worker_pid"]
                run_record["engine_seconds"] = meta["seconds"]
            runs.append(run_record)
            candidate = (image, meta, measured, result_score, dict(current))
            if best is None or result_score["total"] > best[3]["total"]:
                best = candidate
            if result_score["verdict"] == "pass":
                break
            # ADJUST: シードを進め、目標色への寄せを強めて再実行
            current["seed"] = int(current["seed"]) + 101
            current["palette_boost"] = min(current["palette_boost"] + 0.5, 1.0)

        image, meta, measured, result_score, used_params = best
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # COMMIT (保存 + ハッシュチェーン)
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "prompt_ast": prompt_compiler.ast_to_dict(root),
            "size": size,
            "style": style,
            "seed": seed,
            "engine": meta["engine"],
            "analysis": measured,
            "score": result_score,
            "verdict": result_score["verdict"],
            "iterations": len(runs),
            "runs": runs,
        }
        saved = store.add_generation(record, image_bytes)
        return {
            "id": saved["id"],
            "image_url": f"/api/gallery/{saved['id']}/image",
            "prompt": prompt,
            "size": size,
            "style": style,
            "seed": seed,
            "engine": meta["engine"],
            "engine_meta": meta,
            "analysis": measured,
            "score": result_score,
            "verdict": result_score["verdict"],
            "iterations": len(runs),
            "runs": runs,
            "hash": saved["hash"],
            "image_sha256": saved["image_sha256"],
        }
