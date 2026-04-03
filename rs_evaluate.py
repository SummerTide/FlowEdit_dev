"""Evaluate generated RS images: FID, LPIPS, SSIM, PSNR.

Usage:
    python rs_evaluate.py \
        --results_csv ./outputs/rs_flowedit/results.csv \
        --output_dir ./outputs/rs_flowedit/eval
"""

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_image_tensor(path, size=None):
    """Load image as tensor in [0, 1] range."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return transforms.ToTensor()(img)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute PSNR between two image tensors in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM between two image tensors in [0, 1]. Simple per-channel mean."""
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()


def compute_lpips_score(img1: torch.Tensor, img2: torch.Tensor, lpips_fn) -> float:
    """Compute LPIPS between two image tensors in [0, 1]."""
    # LPIPS expects [-1, 1]
    return lpips_fn(img1.unsqueeze(0) * 2 - 1, img2.unsqueeze(0) * 2 - 1).item()


def compute_fid(real_dir: str, gen_dir: str, device: torch.device) -> float:
    """Compute FID between two directories of images."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True).to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Load real images
    for fname in sorted(os.listdir(real_dir)):
        if fname.lower().endswith((".png", ".jpg", ".tif")):
            img = transform(Image.open(os.path.join(real_dir, fname)).convert("RGB"))
            fid.update(img.unsqueeze(0).to(device), real=True)

    # Load generated images
    for fname in sorted(os.listdir(gen_dir)):
        if fname.lower().endswith((".png", ".jpg", ".tif")):
            img = transform(Image.open(os.path.join(gen_dir, fname)).convert("RGB"))
            fid.update(img.unsqueeze(0).to(device), real=False)

    return fid.compute().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True, help="Path to results.csv from rs_inference.py")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--compute_fid", action="store_true", help="Compute FID (requires many samples)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_csv)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load results
    with open(args.results_csv, "r") as f:
        results = list(csv.DictReader(f))

    print(f"Evaluating {len(results)} pairs...")

    # Per-pair metrics
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    # Lazy load LPIPS
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    except ImportError:
        print("Warning: torchmetrics not available, skipping LPIPS")
        lpips_fn = None

    for row in results:
        real_img = load_image_tensor(row["post_img_real"], size=1024).to(device)
        gen_img = load_image_tensor(row["post_img_generated"], size=1024).to(device)

        psnr_scores.append(compute_psnr(gen_img, real_img))
        ssim_scores.append(compute_ssim(gen_img.cpu(), real_img.cpu()))

        if lpips_fn is not None:
            lpips_scores.append(compute_lpips_score(gen_img, real_img, lpips_fn))

    # Aggregate
    report_lines = [
        f"Evaluation Report ({len(results)} pairs)",
        f"{'='*40}",
        f"PSNR:  {np.mean(psnr_scores):.2f} +/- {np.std(psnr_scores):.2f}",
        f"SSIM:  {np.mean(ssim_scores):.4f} +/- {np.std(ssim_scores):.4f}",
    ]

    if lpips_scores:
        report_lines.append(f"LPIPS: {np.mean(lpips_scores):.4f} +/- {np.std(lpips_scores):.4f}")

    # FID (optional, needs enough samples)
    if args.compute_fid:
        # Prepare temp dirs with symlinks
        real_tmp = os.path.join(args.output_dir, "fid_real")
        gen_tmp = os.path.join(args.output_dir, "fid_gen")
        os.makedirs(real_tmp, exist_ok=True)
        os.makedirs(gen_tmp, exist_ok=True)

        for row in results:
            stem = row["stem"]
            real_src = row["post_img_real"]
            gen_src = row["post_img_generated"]
            real_dst = os.path.join(real_tmp, f"{stem}.png")
            gen_dst = os.path.join(gen_tmp, f"{stem}.png")
            if not os.path.exists(real_dst):
                os.symlink(os.path.abspath(real_src), real_dst)
            if not os.path.exists(gen_dst):
                os.symlink(os.path.abspath(gen_src), gen_dst)

        fid_score = compute_fid(real_tmp, gen_tmp, device)
        report_lines.append(f"FID:   {fid_score:.2f}")

    report = "\n".join(report_lines)
    print(report)

    # Save report
    report_path = os.path.join(args.output_dir, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
