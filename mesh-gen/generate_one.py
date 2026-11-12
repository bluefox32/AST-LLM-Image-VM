"""Generate ONE frame: depth guide -> anime base + ControlNet-Depth (CPU).
v2 changes vs v1:
- Base model: AbyssOrangeMix2 SFW (Civitai model version 5021, pruned fp16,
  VAE baked in, ~2.0 GB) downloaded from Civitai and loaded via
  from_single_file. The download endpoint answers anonymously (307 to CDN),
  so no token/secret is needed. If the single-file load fails, the error is
  recorded in info.json and the run falls back to the HF SD1.5 base (v1
  behaviour) so frames are still produced.
- Safety checker disabled: in v1 the 90/270-degree frames came out pure
  black in two independent runs (deterministic), consistent with a
  safety-checker false positive on profile views.
- Per-angle prompt: back-hemisphere angles (135/180/225) drop the face
  wording and say "seen from behind" (v1 drew a face on the back view).
Same seed/steps as v1 otherwise, so differences are attributable.
"""
import argparse, json, os, time, urllib.request

CIVITAI_VERSION = 5021
CIVITAI_URL = f"https://civitai.com/api/download/models/{CIVITAI_VERSION}"
CKPT_PATH = os.path.join("ckpt", "aom2_sfw_fp16.safetensors")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"}

PROMPT_FRONT = ("chibi anime mascot girl head, glossy green hair, two round hair buns, "
                "cute face, character illustration, dark plain background, centered")
PROMPT_BACK = ("back of a chibi anime mascot girl head, glossy green hair, two round hair buns, "
               "seen from behind, character illustration, dark plain background, centered")
NEG = "blurry, deformed, extra limbs, text, watermark, realistic, photo"

def prompt_for(index):
    return PROMPT_BACK if index in (3, 4, 5) else PROMPT_FRONT

def download_ckpt():
    if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1_000_000_000:
        return CKPT_PATH
    os.makedirs("ckpt", exist_ok=True)
    req = urllib.request.Request(CIVITAI_URL, headers=UA)
    with urllib.request.build_opener().open(req, timeout=120) as r, open(CKPT_PATH, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return CKPT_PATH

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--guide", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        print("DRY RUN", json.dumps(vars(args)))
        print("guide exists:", os.path.exists(args.guide))
        print("prompt:", prompt_for(args.index))
        print("base: civitai version", CIVITAI_VERSION)
        return
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from PIL import Image
    t0 = time.time()
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float32)
    base_note = f"civitai:{CIVITAI_VERSION}"
    try:
        ckpt = download_ckpt()
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            ckpt, controlnet=controlnet, torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False)
    except Exception as ex:
        base_note = f"fallback HF SD1.5 (single-file failed: {type(ex).__name__}: {ex})"
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet,
            torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    load_s = time.time() - t0
    guide = Image.open(args.guide).convert("RGB").resize((args.size, args.size))
    t1 = time.time()
    out = pipe(prompt_for(args.index), negative_prompt=NEG, image=guide,
               num_inference_steps=args.steps, guidance_scale=7.0,
               controlnet_conditioning_scale=0.9,
               generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
    gen_s = time.time() - t1
    os.makedirs(args.out_dir, exist_ok=True)
    out.save(os.path.join(args.out_dir, f"frame_{args.index:02d}.png"))
    import diffusers, transformers
    info = {"index": args.index, "angle_deg": args.index * 45, "size": args.size,
            "steps": args.steps, "seed": args.seed, "base": base_note,
            "safety_checker": None, "prompt_kind": "back" if args.index in (3, 4, 5) else "front",
            "load_seconds": round(load_s, 1), "gen_seconds": round(gen_s, 1),
            "torch": torch.__version__, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__}
    with open(os.path.join(args.out_dir, f"info_{args.index:02d}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("DONE", json.dumps(info))

if __name__ == "__main__":
    main()
