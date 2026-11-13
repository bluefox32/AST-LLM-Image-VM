"""通常生成の実用検証 (バッテリ plain_generation_v1) の実行入口。

Flask サーバを介さず、製品スタックそのまま (ImageVM: PARSE → VALIDATE →
COMPILE → GEN → ANALYZE → SCORE → COMMIT) を指定エンジンで1シード分だけ
回し、成果画像と測定値を out ディレクトリへ書き出す。GitHub Actions の
plain-gen ワークフローから呼ばれる想定。

使い方:
  python3 run_plain_check.py --seed 1234 [--engine sd|dummy] [--out-dir plain_out]

バッテリ条件 (BATTERY_plain_generation_v1): ガイド/ControlNet 無しの通常生成。
プロンプト・サイズ・steps・ネガティブは mesh-gen v2 と同一に固定し、差分を
「ガイド層の有無」だけにする。steps/guidance/ネガティブは SDEngine 側の
構築時設定 (engines.SDEngine の既定値)。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "vendor"))
sys.path.insert(0, str(BASE))

PROMPT_FRONT = (
    "chibi anime mascot girl head, glossy green hair, two round hair buns, "
    "cute face, character illustration, dark plain background, centered"
)
SIZE = "512x512"
STYLE = "vivid"  # ImageVM.run の既定 (製品形のまま)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--engine", choices=["sd", "dummy"], default="sd")
    ap.add_argument("--out-dir", default="plain_out")
    args = ap.parse_args()

    import engines
    import prompt_compiler
    import store
    from vm import ImageVM

    # エンジンに実際に渡る compiled params を記録用に得る (compile は決定的で、
    # VM 内部でも同じ呼び出しが行われる)
    _, params = prompt_compiler.compile_prompt(PROMPT_FRONT, SIZE, STYLE, args.seed)

    engine = engines.SDEngine() if args.engine == "sd" else engines.DummyEngine()
    vm = ImageVM(engine=engine)
    t0 = time.time()
    result = vm.run(PROMPT_FRONT, size=SIZE, style=STYLE, seed=args.seed)
    wall = round(time.time() - t0, 1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = store.image_path(result["id"])
    png_name = f"plain_{args.seed}.png"
    if src is not None:
        shutil.copyfile(src, out / png_name)
    summary = {
        "battery": "plain_generation_v1",
        "battery_seed": args.seed,
        "engine_requested": args.engine,
        "prompt_input": PROMPT_FRONT,
        "compiled_prompt": params.get("prompt"),
        "compiled_params": {
            k: params[k] for k in ("size", "style", "seed") if k in params
        },
        "wall_seconds": wall,
        "vm_result": {
            k: result[k]
            for k in (
                "id",
                "engine",
                "engine_meta",
                "score",
                "verdict",
                "iterations",
                "runs",
                "image_sha256",
            )
        },
        "png": png_name if src is not None else None,
    }
    (out / f"plain_{args.seed}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(
        "DONE",
        json.dumps(
            {
                "seed": args.seed,
                "wall_seconds": wall,
                "iterations": result["iterations"],
                "verdict": result["verdict"],
            }
        ),
    )


if __name__ == "__main__":
    main()
