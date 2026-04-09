import argparse
import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from nets.unet import Unet


def normalize_16bit_image(img: np.ndarray, low_percent: float = 1, high_percent: float = 99.5) -> np.ndarray:
    img = img.astype(np.float32)
    low = np.percentile(img, low_percent)
    high = np.percentile(img, high_percent)
    img = np.clip(img, low, high)
    return (img - low) / (high - low + 1e-8)


def load_name_list(dataset_root: str, split: str) -> List[str]:
    split_file = os.path.join(dataset_root, "ImageSets", "Segmentation", f"{split}.txt")
    if not os.path.exists(split_file):
        return []
    with open(split_file, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def load_image_and_label(dataset_root: str, name: str):
    image = np.array(Image.open(os.path.join(dataset_root, "Images", f"{name}.png")))
    label = np.array(Image.open(os.path.join(dataset_root, "Labels", f"{name}.png")))
    if image.ndim == 3:
        image = image[..., 0]
    label = (label > 0).astype(np.uint8)
    image = normalize_16bit_image(image)
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    return image, label


def colorize_heatmap(freq: np.ndarray) -> np.ndarray:
    heat_u8 = np.clip(freq * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)


def main():
    parser = argparse.ArgumentParser(description="Build stable noise pseudo labels from branch-A predictions.")
    parser.add_argument("--dataset-root", type=str, default="Medical_Datasets")
    parser.add_argument("--weights", type=str, required=True, help="Path to branch-A checkpoint.")
    parser.add_argument("--backbone", type=str, default="myunet")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--pred-thr", type=float, default=0.8, help="Foreground prob threshold to mark candidate FP.")
    parser.add_argument("--stable-thr", type=float, default=0.7, help="Frequency threshold for stable noise map.")
    parser.add_argument("--fg-exclude-thr", type=float, default=0.3, help="Exclude pixels frequently in GT foreground.")
    parser.add_argument("--noise-dir", type=str, default="NoiseLabels")
    parser.add_argument("--hotmap-path", type=str, default="logs/noise_hotmap.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    names = load_name_list(args.dataset_root, "train") + load_name_list(args.dataset_root, "val")
    names = sorted(list(set(names)))
    if len(names) == 0:
        raise RuntimeError("No names found in train/val splits.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = Unet(num_classes=args.num_classes, pretrained=False, backbone=args.backbone).to(device).eval()
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt, strict=False)

    fp_count_map = None
    fg_count_map = None
    fp_masks = {}

    with torch.no_grad():
        for name in tqdm(names, desc="Infer branch-A and collect FP"):
            image, gt = load_image_and_label(args.dataset_root, name)
            image = image.to(device)

            outputs = model(image)
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            prob = torch.softmax(logits, dim=1)[:, 1:2, ...]
            prob = prob.squeeze(0).squeeze(0).cpu().numpy()

            if prob.shape != gt.shape:
                prob_t = torch.from_numpy(prob).float().unsqueeze(0).unsqueeze(0)
                prob = F.interpolate(
                    prob_t, size=gt.shape, mode="bilinear", align_corners=False
                ).squeeze().numpy()

            pred_fg = (prob >= args.pred_thr).astype(np.uint8)
            fp_mask = ((pred_fg == 1) & (gt == 0)).astype(np.uint8)

            if fp_count_map is None:
                fp_count_map = np.zeros_like(fp_mask, dtype=np.float32)
                fg_count_map = np.zeros_like(gt, dtype=np.float32)
            fp_count_map += fp_mask
            fg_count_map += gt
            fp_masks[name] = fp_mask

    n_images = float(len(names))
    fp_freq = fp_count_map / n_images
    fg_freq = fg_count_map / n_images

    stable_noise_map = (fp_freq >= args.stable_thr).astype(np.uint8)
    stable_noise_map = (stable_noise_map & (fg_freq < args.fg_exclude_thr).astype(np.uint8)).astype(np.uint8)

    noise_dir = os.path.join(args.dataset_root, args.noise_dir)
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.hotmap_path), exist_ok=True)

    for name in tqdm(names, desc="Save per-image noise pseudo labels"):
        noise_label = (fp_masks[name] & stable_noise_map).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(noise_dir, f"{name}.png"), noise_label)

    cv2.imwrite(os.path.join(noise_dir, "stable_noise_map.png"), stable_noise_map * 255)
    hotmap_bgr = colorize_heatmap(fp_freq)
    cv2.imwrite(args.hotmap_path, hotmap_bgr)

    print(f"Done. N_images={len(names)}")
    print(f"Noise labels dir: {noise_dir}")
    print(f"Stable noise map: {os.path.join(noise_dir, 'stable_noise_map.png')}")
    print(f"Noise hotmap: {args.hotmap_path}")


if __name__ == "__main__":
    main()