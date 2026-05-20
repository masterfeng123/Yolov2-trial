"""
Darknet YOLOv2 weight file loader with batch-normalization folding.

Darknet .weights binary layout (all float32, little-endian):
  Header: [major(i32), minor(i32), revision(i32), seen(i64)]  = 20 bytes
  Layers in order (conv only; pool/reorg have no weights):
    If has_batch_norm:
      bn_biases    [OC]
      bn_weights   [OC]   ← gamma
      bn_means     [OC]
      bn_variances [OC]
      conv_weights [OC][IC][KH][KW]
    Else (last detection conv, no BN):
      conv_biases  [OC]
      conv_weights [OC][IC][KH][KW]

After BN-fold:
  w_folded  = w * gamma / sqrt(var + eps)          shape [OC,IC,KH,KW]
  b_folded  = beta - gamma * mean / sqrt(var + eps) shape [OC]
"""

import struct
import numpy as np
from pathlib import Path

BN_EPS = 1e-5

# YOLOv2-VOC layer topology: (out_c, in_c, ksize, has_bn)
LAYER_CFG = [
    ( 32,    3, 3, True ),   # 0
    ( 64,   32, 3, True ),   # 2
    (128,   64, 3, True ),   # 4
    ( 64,  128, 1, True ),   # 5
    (128,   64, 3, True ),   # 6
    (256,  128, 3, True ),   # 8
    (128,  256, 1, True ),   # 9
    (256,  128, 3, True ),   # 10
    (512,  256, 3, True ),   # 12
    (256,  512, 1, True ),   # 13
    (512,  256, 3, True ),   # 14
    (256,  512, 1, True ),   # 15
    (512,  256, 3, True ),   # 16  ← passthrough
    (1024, 512, 3, True ),   # 18
    ( 512,1024, 1, True ),   # 19
    (1024, 512, 3, True ),   # 20
    ( 512,1024, 1, True ),   # 21
    (1024, 512, 3, True ),   # 22
    (1024,1024, 3, True ),   # 23
    (1024,1024, 3, True ),   # 24
    (1024,3072, 3, True ),   # 26  (after concat: 1024 + 2048 = 3072 in)
    ( 125,1024, 1, False),   # 27  ← no BN, linear
]


def load_weights(path: str) -> list:
    """
    Load a yolov2-voc.weights (or yolov2.weights) file.
    Returns list of dicts:
      { 'weights': np.ndarray [OC,IC,KH,KW],
        'biases':  np.ndarray [OC] }
    in LAYER_CFG order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    with open(path, 'rb') as f:
        # Read header
        major, minor, revision = struct.unpack('<3i', f.read(12))
        # 'seen' may be 4 or 8 bytes depending on version
        seen_raw = f.read(8)
        seen = struct.unpack('<Q', seen_raw)[0]
        print(f"[weights] version {major}.{minor}.{revision}  seen={seen}")

        layers = []
        for (oc, ic, ks, has_bn) in LAYER_CFG:
            n_wt = oc * ic * ks * ks

            if has_bn:
                bn_b = np.frombuffer(f.read(oc * 4), dtype=np.float32).copy()
                bn_g = np.frombuffer(f.read(oc * 4), dtype=np.float32).copy()
                bn_m = np.frombuffer(f.read(oc * 4), dtype=np.float32).copy()
                bn_v = np.frombuffer(f.read(oc * 4), dtype=np.float32).copy()
                w    = np.frombuffer(f.read(n_wt * 4), dtype=np.float32).copy()
                w    = w.reshape(oc, ic, ks, ks)

                # Fold BN
                scale      = bn_g / np.sqrt(bn_v + BN_EPS)      # [OC]
                w_folded   = w * scale[:, None, None, None]      # broadcast
                b_folded   = bn_b - bn_g * bn_m / np.sqrt(bn_v + BN_EPS)
            else:
                b_folded = np.frombuffer(f.read(oc * 4), dtype=np.float32).copy()
                w        = np.frombuffer(f.read(n_wt * 4), dtype=np.float32).copy()
                w_folded = w.reshape(oc, ic, ks, ks)

            layers.append({'weights': w_folded, 'biases': b_folded})
            print(f"  layer {len(layers):2d}: OC={oc:4d} IC={ic:4d} K={ks} "
                  f"BN={has_bn}  w_range=[{w_folded.min():.3f},{w_folded.max():.3f}]")

        remaining = len(f.read())
        if remaining > 0:
            print(f"  WARNING: {remaining} bytes unread in weight file")

    return layers


def make_synthetic_weights(seed: int = 42) -> list:
    """
    Generate small random weights in the same format as load_weights().
    Used when the real .weights file is unavailable (CI / unit testing).
    """
    rng = np.random.default_rng(seed)
    layers = []
    for (oc, ic, ks, _) in LAYER_CFG:
        # Xavier initialisation scale
        fan_in = ic * ks * ks
        std    = np.sqrt(2.0 / fan_in)
        w = rng.normal(0, std, (oc, ic, ks, ks)).astype(np.float32)
        b = np.zeros(oc, dtype=np.float32)
        layers.append({'weights': w, 'biases': b})
    return layers
