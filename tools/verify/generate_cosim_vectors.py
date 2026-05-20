"""
Generate binary test vectors for HLS co-simulation.

For each testable accelerator command (CONV / POOL / REORG), this script:
  1. Runs the YOLOv2 reference forward pass (float32 + quantised).
  2. Writes input.bin, weight.bin, expected.bin as int16 files
     in  hls/tb/cosim_data/<layer_idx>/

The HLS testbench tb_cosim.cpp reads these files, calls yolov2_accel,
then reads back the hardware output for comparison.

Layout of each .bin file:  little-endian int16, row-major, channel-first
  input.bin   : [in_c, in_h, in_w]
  weight.bin  : [out_c, in_c, ksize, ksize] + [out_c] biases
  expected.bin: [out_c, out_h, out_w]
"""

import numpy as np
import struct
import sys
from pathlib import Path

ROOT     = Path(__file__).parent.parent.parent
VEC_DIR  = ROOT / "hls" / "tb" / "cosim_data"
VERIFY   = Path(__file__).parent

sys.path.insert(0, str(VERIFY))
from quantize_utils import to_int16, to_fixed, write_bin
from darknet_weights import make_synthetic_weights, LAYER_CFG
from yolov2_model     import YOLOv2, maxpool, reorg
from generate_test_image import prepare_test_image

# ---------------------------------------------------------------------------
# Layer index mapping:  model layer → LAYER_CFG index
# The 28 model layers map to convolution entries in LAYER_CFG as follows:
# (pool and reorg layers have no weights)
# ---------------------------------------------------------------------------
MODEL_TO_CONV = {
     0: 0,   # conv 0
     2: 1,   # conv 2
     4: 2,   # conv 4
     5: 3,   # conv 5
     6: 4,   # conv 6
     8: 5,   # conv 8
     9: 6,   # conv 9
    10: 7,   # conv 10
    12: 8,   # conv 12
    13: 9,   # conv 13
    14:10,   # conv 14
    15:11,   # conv 15
    16:12,   # conv 16
    18:13,   # conv 18
    19:14,   # conv 19
    20:15,   # conv 20
    21:16,   # conv 21
    22:17,   # conv 22
    23:18,   # conv 23
    24:19,   # conv 24
    26:20,   # conv 26
    27:21,   # conv 27
}

# Subset of layers we generate vectors for (avoids very large files)
VECTOR_LAYERS = [0, 1, 2, 7, 11, 16, 17, 24, 25, 26, 27]


def save_layer(layer_idx: int, cmd: str,
               inp: np.ndarray, expected: np.ndarray,
               weights: np.ndarray = None, biases: np.ndarray = None):
    """Save test vectors to cosim_data/<layer_idx>/"""
    d = VEC_DIR / str(layer_idx)
    d.mkdir(parents=True, exist_ok=True)

    write_bin(str(d / "input.bin"),    inp)
    write_bin(str(d / "expected.bin"), expected)

    if weights is not None and biases is not None:
        wt_flat = np.concatenate([weights.flatten(), biases.flatten()])
        write_bin(str(d / "weight.bin"), wt_flat)

    # Write metadata (JSON-like simple format read by the HLS testbench)
    in_c, in_h, in_w = inp.shape
    out_shape = expected.shape
    ks = weights.shape[2] if weights is not None else 0

    meta = {
        "cmd":      cmd,
        "in_h":     in_h,  "in_w": in_w,   "in_c": in_c,
        "out_c":    out_shape[0] if len(out_shape)==3 else in_c,
        "out_h":    out_shape[1] if len(out_shape)==3 else in_h//2,
        "out_w":    out_shape[2] if len(out_shape)==3 else in_w//2,
        "ksize":    ks,
        "stride":   1 if cmd == "CONV" else 2,
        "pad":      1 if (cmd == "CONV" and ks == 3) else 0,
        "flags":    1 if (cmd == "CONV" and layer_idx != 27) else 0,
    }
    with open(d / "meta.txt", "w") as f:
        for k, v in meta.items():
            f.write(f"{k}={v}\n")

    in_kb  = inp.size * 2 / 1024
    out_kb = expected.size * 2 / 1024
    print(f"  [L{layer_idx:2d}] {cmd:5s}  in={inp.shape}  out={expected.shape}  "
          f"in={in_kb:.1f}KB  out={out_kb:.1f}KB  → {d}")


def generate_all(weights_path: str = None):
    print("=" * 60)
    print("Generating HLS co-simulation test vectors")
    print("=" * 60)

    # Load weights
    if weights_path and Path(weights_path).exists():
        from darknet_weights import load_weights
        layers = load_weights(weights_path)
        print(f"[weights] loaded from {weights_path}")
    else:
        print("[weights] using synthetic random weights (for structure verification)")
        layers = make_synthetic_weights()

    # Prepare test image
    img_path = VERIFY / "test_image_rgb.npy"
    if img_path.exists():
        img = np.load(img_path)
        print(f"[image]   loaded  {img_path}")
    else:
        img = prepare_test_image()

    # Run quantised forward pass and capture per-layer outputs
    print("\nRunning quantised reference forward pass ...")
    model_q = YOLOv2(layers, quantised=True)
    out_q   = model_q.forward(img)
    print(f"  Final output shape: {out_q.shape}  "
          f"range=[{float(out_q.min()):.3f}, {float(out_q.max()):.3f}]")

    # Also run float32 for comparison stats
    print("Running float32 reference forward pass ...")
    model_f = YOLOv2(layers, quantised=False)
    out_f   = model_f.forward(img)

    print(f"\nGenerating vectors for layers: {VECTOR_LAYERS}")

    # ---- LAYER 0: first conv (small input, good for sanity check) --------
    if 0 in VECTOR_LAYERS:
        inp_q = to_fixed(img)
        ci    = MODEL_TO_CONV[0]
        w_q   = to_fixed(layers[ci]['weights'])
        b_q   = to_fixed(layers[ci]['biases'])
        exp_q = model_q.layer_outputs[0]
        save_layer(0, "CONV", inp_q, exp_q, w_q, b_q)

    # ---- LAYER 1: first maxpool -------------------------------------------
    if 1 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs[0]
        exp_q = model_q.layer_outputs[1]
        save_layer(1, "POOL", inp_q, exp_q)

    # ---- LAYER 2: second conv (64 out) ------------------------------------
    if 2 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs[1]
        ci    = MODEL_TO_CONV[2]
        w_q   = to_fixed(layers[ci]['weights'])
        b_q   = to_fixed(layers[ci]['biases'])
        exp_q = model_q.layer_outputs[2]
        save_layer(2, "CONV", inp_q, exp_q, w_q, b_q)

    # ---- LAYER 7: pool after 128-channel block ----------------------------
    if 7 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs[6]
        exp_q = model_q.layer_outputs[7]
        save_layer(7, "POOL", inp_q, exp_q)

    # ---- LAYER 11: pool 52→26 --------------------------------------------
    if 11 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs[10]
        exp_q = model_q.layer_outputs[11]
        save_layer(11, "POOL", inp_q, exp_q)

    # ---- LAYER 16: passthrough conv (26×26×512) --------------------------
    if 16 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs[15]
        ci    = MODEL_TO_CONV[16]
        w_q   = to_fixed(layers[ci]['weights'])
        b_q   = to_fixed(layers[ci]['biases'])
        exp_q = model_q.layer_outputs[16]
        save_layer(16, "CONV", inp_q, exp_q, w_q, b_q)

    # ---- LAYER 17: pool 26→13 --------------------------------------------
    if 17 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs[16]
        exp_q = model_q.layer_outputs[17]
        save_layer(17, "POOL", inp_q, exp_q)

    # ---- LAYER 24: 1024→1024 conv ----------------------------------------
    if 24 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs[23]
        ci    = MODEL_TO_CONV[24]
        w_q   = to_fixed(layers[ci]['weights'])
        b_q   = to_fixed(layers[ci]['biases'])
        exp_q = model_q.layer_outputs[24]
        save_layer(24, "CONV", inp_q, exp_q, w_q, b_q)

    # ---- LAYER 25: REORG -------------------------------------------------
    if 25 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs['pt']
        exp_q = model_q.layer_outputs[25]
        save_layer(25, "REORG", inp_q, exp_q)

    # ---- LAYER 26: detection conv (3072 in) ------------------------------
    if 26 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs['concat']
        ci    = MODEL_TO_CONV[26]
        w_q   = to_fixed(layers[ci]['weights'])
        b_q   = to_fixed(layers[ci]['biases'])
        exp_q = model_q.layer_outputs[26]
        save_layer(26, "CONV", inp_q, exp_q, w_q, b_q)

    # ---- LAYER 27: final detection conv (linear output) ------------------
    if 27 in VECTOR_LAYERS:
        inp_q = model_q.layer_outputs[26]
        ci    = MODEL_TO_CONV[27]
        w_q   = to_fixed(layers[ci]['weights'])
        b_q   = to_fixed(layers[ci]['biases'])
        exp_q = out_q
        save_layer(27, "CONV", inp_q, exp_q, w_q, b_q)

    # Save full output for detection decode verification
    np.save(VERIFY / "yolov2_output_quant.npy", out_q)
    np.save(VERIFY / "yolov2_output_float.npy", out_f)

    print(f"\n[done] Vectors written to {VEC_DIR}")
    print(f"       Float output  → {VERIFY/'yolov2_output_float.npy'}")
    print(f"       Quant output  → {VERIFY/'yolov2_output_quant.npy'}")
    return model_q, model_f


if __name__ == "__main__":
    wt = sys.argv[1] if len(sys.argv) > 1 else None
    generate_all(wt)
