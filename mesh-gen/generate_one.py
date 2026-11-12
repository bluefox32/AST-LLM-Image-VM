"""Generate ONE frame: depth guide -> SD1.5 + ControlNet-Depth (CPU).
Same prompt/seed as the Colab pilot. Writes frame PNG + info.json with
timing and versions so results from different runners stay comparable.
"""
import argparse, json, os, time

PROMPT = ("chibi anime mascot girl head, glossy green hair, two round hair buns, "
          "cute face, character illustration, dark plain background, centered")
NEG = "blurry, deformed, extra limbs, text, watermark, realistic, photo"

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
    cfg = vars(args)
    if args.dry_run:
        print("DRY RUN", json.dumps(cfg))
        print("guide exists:", os.path.exists(args.guide))
        return
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from PIL import Image
    t0 = time.time()
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    load_s = time.time() - t0
    guide = Image.open(args.guide).convert("RGB").resize((args.size, args.size))
    t1 = time.time()
    out = pipe(PROMPT, negative_prompt=NEG, image=guide, num_inference_steps=args.steps,
               guidance_scale=7.0, controlnet_conditioning_scale=0.9,
               generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
    gen_s = time.time() - t1
    os.makedirs(args.out_dir, exist_ok=True)
    out.save(os.path.join(args.out_dir, f"frame_{args.index:02d}.png"))
    info = {"index": args.index, "angle_deg": args.index * 45, "size": args.size,
            "steps": args.steps, "seed": args.seed, "load_seconds": round(load_s, 1),
            "gen_seconds": round(gen_s, 1), "torch": torch.__version__}
    import diffusers, transformers
    info["diffusers"] = diffusers.__version__
    info["transformers"] = transformers.__version__
    with open(os.path.join(args.out_dir, f"info_{args.index:02d}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("DONE", json.dumps(info))

if __name__ == "__main__":
    main()
