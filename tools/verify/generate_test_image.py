"""
Prepare a 416×416 test image for YOLOv2 verification.

Strategy
--------
1. Try to download a COCO 2017 validation image (dog + bicycle scene).
2. If no internet, generate a synthetic scene with geometric shapes.

Output
------
  test_image_rgb.npy    – float32 [3, 416, 416] in [0,1]  (CHW format)
  test_image.png        – visual PNG for inspection
"""

import numpy as np
from pathlib import Path
import sys

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Synthetic image generator (deterministic, no internet needed)
# ---------------------------------------------------------------------------

def make_synthetic_image(seed: int = 7) -> np.ndarray:
    """
    Return float32 [3, 416, 416] in [0,1].

    Contains:
      - Gradient sky background
      - A green rectangle (simulates a vehicle)
      - A red circle (simulates a traffic light)
      - A person-shaped silhouette (rectangle + ellipse)
      - White noise texture on the ground
    """
    rng = np.random.default_rng(seed)
    H, W = 416, 416
    img = np.zeros((3, H, W), dtype=np.float32)

    # Sky gradient (top half)
    for row in range(H // 2):
        t = row / (H / 2)
        img[0, row, :] = 0.4 + 0.3 * t
        img[1, row, :] = 0.6 + 0.2 * t
        img[2, row, :] = 0.9 - 0.2 * t

    # Ground (bottom half)
    img[0, H//2:, :] = 0.35
    img[1, H//2:, :] = 0.30
    img[2, H//2:, :] = 0.20
    noise = rng.normal(0, 0.04, (3, H//2, W)).astype(np.float32)
    img[:, H//2:, :] += noise

    # Green car rectangle  [y1:y2, x1:x2]
    y1,y2,x1,x2 = 220, 310, 80, 250
    img[0, y1:y2, x1:x2] = img[0, y1:y2, x1:x2] * 0.3
    img[1, y1:y2, x1:x2] = 0.65
    img[2, y1:y2, x1:x2] = img[2, y1:y2, x1:x2] * 0.3

    # Windows on the car
    img[:, y1+10:y1+50, x1+20:x1+70]   = np.array([0.6, 0.8, 0.9])[:, None, None]
    img[:, y1+10:y1+50, x1+90:x1+140]  = np.array([0.6, 0.8, 0.9])[:, None, None]

    # Wheels (dark circles approximated by rectangles)
    img[:, y2-20:y2, x1+20:x1+50]   = np.array([0.1, 0.1, 0.1])[:, None, None]
    img[:, y2-20:y2, x1+130:x1+160] = np.array([0.1, 0.1, 0.1])[:, None, None]

    # Red traffic light circle
    cx, cy, r = 330, 120, 18
    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy)**2 + (xx - cx)**2 <= r**2
    img[0, mask] = 0.9
    img[1, mask] = 0.1
    img[2, mask] = 0.1

    # Person silhouette
    px1, px2 = 310, 340
    # Body
    img[0, 180:290, px1:px2] = 0.8
    img[1, 180:290, px1:px2] = 0.5
    img[2, 180:290, px1:px2] = 0.3
    # Head
    yy2, xx2 = np.ogrid[:H, :W]
    head_mask = (yy2 - 165)**2 + (xx2 - 325)**2 <= 16**2
    img[0, head_mask] = 0.9
    img[1, head_mask] = 0.75
    img[2, head_mask] = 0.6

    # Bicycle frame lines (simplified as thin rectangles)
    grey = np.array([0.5, 0.5, 0.5])[:, None, None]
    img[:, 260:265, 150:310] = grey   # horizontal bar
    img[:, 230:270, 145:155] = grey   # left fork
    img[:, 230:270, 300:310] = grey   # right fork

    # Clamp [0, 1]
    img = np.clip(img, 0.0, 1.0)
    return img


# ---------------------------------------------------------------------------
# COCO download (single image, ~80 KB)
# ---------------------------------------------------------------------------

COCO_URLS = [
    # COCO 2017 val – image with dogs and people (known COCO ID)
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
]


def try_download_coco() -> np.ndarray | None:
    try:
        import requests
        from PIL import Image
        import io
        for url in COCO_URLS:
            try:
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200:
                    pil = Image.open(io.BytesIO(resp.content)).convert('RGB')
                    pil = pil.resize((416, 416), Image.BILINEAR)
                    arr = np.array(pil, dtype=np.float32) / 255.0  # [H,W,3]
                    arr = arr.transpose(2, 0, 1)                   # [3,H,W]
                    print(f"[image] Downloaded COCO image from {url}")
                    return arr
            except Exception:
                continue
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_test_image(force_synthetic: bool = False) -> np.ndarray:
    out_npy  = OUT_DIR / "test_image_rgb.npy"
    out_png  = OUT_DIR / "test_image.png"

    img = None
    if not force_synthetic:
        img = try_download_coco()

    if img is None:
        print("[image] Using synthetic test image")
        img = make_synthetic_image()

    np.save(out_npy, img)

    # Save PNG
    try:
        from PIL import Image
        hwc = (np.clip(img.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(hwc).save(out_png)
        print(f"[image] Saved {out_png}")
    except Exception as e:
        print(f"[image] Could not save PNG: {e}")

    print(f"[image] Saved {out_npy}  shape={img.shape}  "
          f"range=[{img.min():.3f}, {img.max():.3f}]")
    return img


if __name__ == "__main__":
    force = "--synthetic" in sys.argv
    prepare_test_image(force_synthetic=force)
