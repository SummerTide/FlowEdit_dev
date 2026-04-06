# rs_baseline_gemini.py
"""Baseline using Gemini's native image generation for RS change detection.

Sends (pre_image, pre_segmap, post_segmap) to Gemini with a structured prompt,
asking it to generate the post-change aerial image.

Output format matches rs_inference.py for direct comparison with rs_evaluate.py.

Requirements:
    pip install google-genai Pillow

Usage:
    python rs_baseline_gemini.py \
        --hiucd_root /path/to/Hi-UCD \
        --split mini_val \
        --output_dir ./outputs/rs_gemini_baseline \
        --api_key YOUR_GEMINI_API_KEY

    # Or set env var:
    export GEMINI_API_KEY=YOUR_KEY
    python rs_baseline_gemini.py --hiucd_root /path/to/Hi-UCD --split mini_val
"""

import argparse
import base64
import csv
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from rs_data.hiucd import (
    HIUCD_CLASSES,
    hiucd_segmap_to_rgb,
    hiucd_segmap_to_text,
    parse_hiucd_mask,
)


# ---------- Prompt ----------

SYSTEM_PROMPT = """\
You are an expert remote sensing image synthesis system. Your task is to \
generate realistic post-change aerial/satellite imagery based on provided \
inputs. You must output ONLY the generated image with no text."""

COLOR_LEGEND = "\n".join(
    f"- RGB({c['color'][0]},{c['color'][1]},{c['color'][2]}): {c['name']}"
    for c in HIUCD_CLASSES.values()
)

TASK_PROMPT_TEMPLATE = """\
I provide three images for a remote sensing urban change generation task:

**Image 1 — Pre-change aerial photograph (2018):**
A 512×512 aerial image at 0.1m GSD from Tallinn, Estonia.

**Image 2 — Pre-change segmentation map:**
Color-coded land cover map corresponding to Image 1.

**Image 3 — Post-change segmentation map (target layout, 2019):**
Color-coded land cover map showing the desired land cover after urban changes.

**Segmentation map color legend:**
{color_legend}

**Land cover changes between the two segmaps:**
{change_description}

**Task:**
Generate a realistic post-change aerial image (2019) that:
1. Matches the spatial layout defined by Image 3 (post-change segmap)
2. Preserves the visual style, lighting, texture, and resolution of Image 1
3. Keeps unchanged regions identical or near-identical to Image 1
4. Realistically renders newly appeared land cover types (e.g. new buildings, cleared woodland, new roads)

Output ONLY the generated 512×512 aerial image. No text, no borders, no labels."""


def describe_changes(pre_seg_np, post_seg_np):
    """Generate a text description of land cover changes between two segmaps."""
    changes = []
    for idx, info in HIUCD_CLASSES.items():
        if idx == 0:
            continue
        pre_pct = np.mean(pre_seg_np == idx) * 100
        post_pct = np.mean(post_seg_np == idx) * 100
        diff = post_pct - pre_pct
        if abs(diff) > 1.0:
            direction = "increased" if diff > 0 else "decreased"
            changes.append(f"- {info['name']}: {pre_pct:.1f}% → {post_pct:.1f}% ({direction})")
    if not changes:
        return "Minimal changes between the two time periods."
    return "\n".join(changes)


def build_prompt(pre_seg_np, post_seg_np):
    change_desc = describe_changes(pre_seg_np, post_seg_np)
    return TASK_PROMPT_TEMPLATE.format(
        color_legend=COLOR_LEGEND,
        change_description=change_desc,
    )


# ---------- Gemini API ----------

def create_client(api_key):
    """Create Gemini client. Supports both google-genai (new) and google-generativeai (old)."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, "new"
    except ImportError:
        pass

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai, "old"
    except ImportError:
        raise ImportError(
            "Neither 'google-genai' nor 'google-generativeai' is installed.\n"
            "Install with: pip install google-genai"
        )


def call_gemini_new_sdk(client, model_name, prompt, images, max_retries=3):
    """Call Gemini using the new google-genai SDK."""
    from google.genai import types

    contents = [types.Part.from_text(text=prompt)]
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        contents.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            # Extract generated image from response
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    img_data = part.inline_data.data
                    return Image.open(io.BytesIO(img_data)).convert("RGB")
            print(f"  Warning: no image in response (attempt {attempt+1}), retrying...")
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise
    return None


def call_gemini_old_sdk(genai_module, model_name, prompt, images, max_retries=3):
    """Call Gemini using the old google-generativeai SDK."""
    model = genai_module.GenerativeModel(
        model_name,
        system_instruction=SYSTEM_PROMPT,
    )
    contents = [prompt] + images

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                contents,
                generation_config=genai_module.GenerationConfig(
                    response_mime_type="image/png",
                ),
            )
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data.data:
                    return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
            print(f"  Warning: no image in response (attempt {attempt+1}), retrying...")
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise
    return None


def generate_with_gemini(client, sdk_version, model_name, prompt, images):
    if sdk_version == "new":
        return call_gemini_new_sdk(client, model_name, prompt, images)
    else:
        return call_gemini_old_sdk(client, model_name, prompt, images)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Gemini baseline for RS change generation")
    parser.add_argument("--hiucd_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/rs_gemini_baseline")
    parser.add_argument("--split", type=str, default="mini_val")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY env)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-image",
                        help="Gemini model name with image generation support")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls in seconds")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Provide --api_key or set GEMINI_API_KEY environment variable")

    client, sdk_version = create_client(api_key)
    print(f"Using {sdk_version} SDK, model: {args.model}")

    # Collect Hi-UCD pairs
    split_dir = Path(args.hiucd_root) / args.split
    pre_img_dir = split_dir / "image" / "2018"
    post_img_dir = split_dir / "image" / "2019"
    mask_dir = split_dir / "mask" / "2018_2019"

    mask_files = sorted(mask_dir.glob("*.png"))
    if args.max_samples:
        mask_files = mask_files[:args.max_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    failed = []

    for mask_path in tqdm(mask_files, desc="Generating"):
        stem = mask_path.stem
        pre_img_path = pre_img_dir / f"{stem}.png"
        post_img_path = post_img_dir / f"{stem}.png"

        if not pre_img_path.exists() or not post_img_path.exists():
            print(f"Skipping {stem}: missing image files")
            continue

        # Load inputs
        pre_img = Image.open(pre_img_path).convert("RGB")
        mask_rgb = np.array(Image.open(mask_path))
        pre_seg_np, post_seg_np, _ = parse_hiucd_mask(mask_rgb)
        pre_seg_pil = Image.fromarray(hiucd_segmap_to_rgb(pre_seg_np))
        post_seg_pil = Image.fromarray(hiucd_segmap_to_rgb(post_seg_np))

        # Build prompt
        prompt = build_prompt(pre_seg_np, post_seg_np)

        # Call Gemini: [pre_image, pre_segmap, post_segmap]
        gen_img = generate_with_gemini(
            client, sdk_version, args.model,
            prompt, [pre_img, pre_seg_pil, post_seg_pil],
        )

        if gen_img is None:
            print(f"  Failed to generate for {stem}")
            failed.append(stem)
            continue

        # Resize to match original if needed
        orig_size = pre_img.size
        if gen_img.size != orig_size:
            gen_img = gen_img.resize(orig_size, Image.LANCZOS)

        # Save
        out_path = os.path.join(args.output_dir, f"{stem}_generated_post.png")
        gen_img.save(out_path)

        results.append({
            "stem": stem,
            "pre_img": str(pre_img_path),
            "post_img_real": str(post_img_path),
            "post_img_generated": out_path,
            "flowedit_src_prompt": "",
            "flowedit_tar_prompt": hiucd_segmap_to_text(post_seg_np),
            "prompt_mode": "gemini_baseline",
        })

        # Rate limiting
        time.sleep(args.delay)

    # Save results CSV (compatible with rs_evaluate.py)
    manifest_path = os.path.join(args.output_dir, "results.csv")
    if results:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nDone. {len(results)} generated, {len(failed)} failed.")
    if failed:
        print(f"Failed stems: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    print(f"Results: {manifest_path}")


if __name__ == "__main__":
    main()
