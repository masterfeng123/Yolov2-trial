"""
YOLOv2 HLS Accelerator – Complete Verification Suite
=====================================================

Steps executed
--------------
1. Prepare test image (download COCO or generate synthetic).
2. Generate co-simulation test vectors.
3. Run float32 + quantised software inference.
4. Print layer-by-layer comparison report.
5. Render annotated detection visualisation.
6. (Optional) Read HLS co-sim outputs and compare.

Usage
-----
  python3 run_verification.py                        # synthetic weights
  python3 run_verification.py --weights yolov2.weights
  python3 run_verification.py --weights yolov2.weights --hls_cosim
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")    # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT   = Path(__file__).parent.parent.parent
VERIFY = Path(__file__).parent
sys.path.insert(0, str(VERIFY))

from quantize_utils       import layer_stats
from darknet_weights      import make_synthetic_weights
from yolov2_model         import YOLOv2, decode_detections, VOC_CLASSES
from generate_test_image  import prepare_test_image
from generate_cosim_vectors import generate_all
from layer_compare        import (compare_float_vs_quant,
                                   compare_ref_vs_hls,
                                   print_report,
                                   write_html_report)


# ---------------------------------------------------------------------------
# Visualise detections
# ---------------------------------------------------------------------------

COLORS = matplotlib.colormaps.get_cmap('tab20').colors

def draw_detections(img_chw, dets, title, out_path):
    hwc = np.clip(img_chw.transpose(1, 2, 0), 0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(hwc)
    ax.set_title(title, fontsize=13)
    ax.axis('off')
    for d in dets:
        bx, by, bw, bh = d['box']
        x0, y0 = bx - bw/2, by - bh/2
        color  = COLORS[d['class'] % 20]
        rect   = mpatches.Rectangle((x0, y0), bw, bh,
                                     linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label = f"{d['label']} {d['conf']:.2f}"
        ax.text(x0, y0 - 4, label, fontsize=9,
                color='white', backgroundcolor=color,
                bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  [viz] {out_path}")


# ---------------------------------------------------------------------------
# Layer statistics plot
# ---------------------------------------------------------------------------

def plot_layer_stats(rows, title, out_path):
    layers = [str(r['layer']) for r in rows if 'mae' in r]
    maes   = [r['mae']     for r in rows if 'mae' in r]
    snrs   = [min(r['snr_db'], 60) for r in rows if 'mae' in r]

    if not layers:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(layers)*0.6), 7))
    x = np.arange(len(layers))

    ax1.bar(x, maes, color='steelblue', alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.set_ylabel("Mean Abs Error"); ax1.set_title(title)
    ax1.axhline(1/256, color='red', linestyle='--', label='1 LSB')
    ax1.legend()

    ax2.bar(x, snrs, color='darkorange', alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(layers, rotation=45, ha='right')
    ax2.set_ylabel("SNR (dB)")
    ax2.axhline(20, color='red', linestyle='--', label='20 dB threshold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  [viz] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="YOLOv2 HLS verification suite")
    ap.add_argument("--weights",   default=None,  help="Path to yolov2-voc.weights")
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic test image")
    ap.add_argument("--hls_cosim", action="store_true", help="Compare against HLS co-sim outputs")
    ap.add_argument("--thresh",    type=float, default=0.25, help="Detection threshold")
    args = ap.parse_args()

    print("=" * 65)
    print("  YOLOv2 HLS Accelerator – Verification Suite")
    print("=" * 65)

    # ---- 1. Test image -------------------------------------------------------
    print("\n[1/5] Preparing test image ...")
    img_path = VERIFY / "test_image_rgb.npy"
    if img_path.exists():
        img = np.load(img_path)
        print(f"      Loaded cached {img_path}")
    else:
        img = prepare_test_image(force_synthetic=args.synthetic)

    # ---- 2. Weights ----------------------------------------------------------
    print("\n[2/5] Loading weights ...")
    if args.weights and Path(args.weights).exists():
        from darknet_weights import load_weights
        layers = load_weights(args.weights)
        wt_src = "real"
    else:
        layers = make_synthetic_weights()
        wt_src = "synthetic"
    print(f"      Source: {wt_src}  ({len(layers)} conv layers)")

    # ---- 3. Forward passes ---------------------------------------------------
    print("\n[3/5] Running reference forward passes ...")

    t0 = time.time()
    model_f = YOLOv2(layers, quantised=False)
    out_f   = model_f.forward(img)
    t_float = time.time() - t0
    print(f"      Float32  : {t_float:.2f}s  out={out_f.shape}")

    t0 = time.time()
    model_q = YOLOv2(layers, quantised=True)
    out_q   = model_q.forward(img)
    t_quant = time.time() - t0
    print(f"      Quantised: {t_quant:.2f}s  out={out_q.shape}")

    # ---- 4. Layer comparison -------------------------------------------------
    print("\n[4/5] Layer-by-layer comparison ...")
    fq_rows = compare_float_vs_quant(model_q, model_f)
    print_report(fq_rows, "Float32 vs Quantised (ap_fixed<16,8>)")

    hls_rows = []
    if args.hls_cosim:
        hls_rows = compare_ref_vs_hls()
        print_report(hls_rows, "Quantised Reference vs HLS Co-sim")

    # ---- 5. Detections + visualisation ---------------------------------------
    print("\n[5/5] Detection decode & visualisation ...")

    if wt_src == "real":
        dets_f = decode_detections(out_f, thresh=args.thresh)
        dets_q = decode_detections(out_q, thresh=args.thresh)

        print(f"  Float32  detections: {len(dets_f)}")
        for d in dets_f:
            print(f"    {d['label']:15s}  conf={d['conf']:.3f}  "
                  f"box=[{d['box'][0]:.0f},{d['box'][1]:.0f},"
                  f"{d['box'][2]:.0f},{d['box'][3]:.0f}]")

        print(f"  Quantised detections: {len(dets_q)}")
        for d in dets_q:
            print(f"    {d['label']:15s}  conf={d['conf']:.3f}  "
                  f"box=[{d['box'][0]:.0f},{d['box'][1]:.0f},"
                  f"{d['box'][2]:.0f},{d['box'][3]:.0f}]")

        draw_detections(img, dets_f, "YOLOv2 Float32 Detections",
                        VERIFY / "detections_float.png")
        draw_detections(img, dets_q, "YOLOv2 Quantised Detections",
                        VERIFY / "detections_quant.png")
    else:
        print("  (skip detection decode: synthetic weights produce meaningless boxes)")
        # Still save the image so the pipeline is testable
        from PIL import Image as PILImage
        hwc = (np.clip(img.transpose(1,2,0),0,1)*255).astype(np.uint8)
        PILImage.fromarray(hwc).save(VERIFY / "test_image_vis.png")
        print(f"  Saved test_image_vis.png")

    # Stat plots (always)
    plot_layer_stats(fq_rows, "Float32 vs Quantised per layer",
                     VERIFY / "layer_stats_fq.png")
    if hls_rows:
        plot_layer_stats(hls_rows, "Quantised vs HLS Co-sim per layer",
                         VERIFY / "layer_stats_hls.png")

    # HTML report
    write_html_report(fq_rows, hls_rows, VERIFY / "verification_report.html")

    # ---- Generate co-sim vectors for use with tb_cosim.cpp -----------------
    print("\n[+] Generating HLS co-simulation test vectors ...")
    generate_all(args.weights if wt_src == "real" else None)

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Verification complete!")
    print("=" * 65)
    print(f"  test image      : tools/verify/test_image.png")
    print(f"  layer stats plot: tools/verify/layer_stats_fq.png")
    print(f"  HTML report     : tools/verify/verification_report.html")
    print(f"  cosim vectors   : hls/tb/cosim_data/<layer>/")
    if wt_src == "real":
        print(f"  detections      : tools/verify/detections_float.png")
        print(f"                    tools/verify/detections_quant.png")
    print()


if __name__ == "__main__":
    main()
