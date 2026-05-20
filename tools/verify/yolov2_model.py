"""
NumPy reference implementation of YOLOv2 (28-layer Darknet-19, VOC 20-class).

Each layer's output is saved in self.layer_outputs for validation.
Both float32 and ap_fixed<16,8> quantised modes are supported.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from quantize_utils import to_fixed, to_int16, from_int16, SCALE

LEAKY = 0.1


# ---------------------------------------------------------------------------
# Core kernels (vectorised NumPy)
# ---------------------------------------------------------------------------

def _pad(x, p):
    if p <= 0:
        return x
    return np.pad(x, ((0,0), (p,p), (p,p)), mode='constant')


def conv2d(x, w, b, stride=1, pad=0, quantised=False):
    """
    x : float32 [IC, IH, IW]
    w : float32 [OC, IC, KH, KW]
    b : float32 [OC]
    returns: float32 [OC, OH, OW]

    Uses im2col for efficiency.
    """
    IC, IH, IW = x.shape
    OC, _IC, KH, KW = w.shape
    assert IC == _IC

    xp = _pad(x, pad)
    _, PH, PW = xp.shape
    OH = (PH - KH) // stride + 1
    OW = (PW - KW) // stride + 1

    # im2col via stride tricks: shape [OH, OW, IC, KH, KW]
    s = xp.strides
    cols = as_strided(
        xp,
        shape=(OH, OW, IC, KH, KW),
        strides=(s[1]*stride, s[2]*stride, s[0], s[1], s[2]),
        writeable=False
    )
    # Reshape to [OH*OW, IC*KH*KW]
    cols_flat = cols.reshape(OH * OW, IC * KH * KW)
    w_flat    = w.reshape(OC, IC * KH * KW)          # [OC, IC*KH*KW]

    if quantised:
        cols_flat = to_fixed(cols_flat)
        w_flat    = to_fixed(w_flat)
        b         = to_fixed(b)

    # Matrix multiply + bias: [OH*OW, OC]
    out = cols_flat @ w_flat.T + b                     # broadcast bias

    if quantised:
        out = to_fixed(out)

    return out.T.reshape(OC, OH, OW).astype(np.float32)


def leaky_relu(x, quantised=False):
    out = np.where(x >= 0, x, LEAKY * x)
    if quantised:
        out = to_fixed(out)
    return out.astype(np.float32)


def maxpool(x, ksize=2, stride=2):
    """x: [C, H, W]  →  [C, H//stride, W//stride]"""
    C, H, W = x.shape
    OH = H // stride
    OW = W // stride
    # View with stride tricks for the 2×2 window
    s = x.strides
    windows = as_strided(
        x,
        shape=(C, OH, OW, ksize, ksize),
        strides=(s[0], s[1]*stride, s[2]*stride, s[1], s[2]),
        writeable=False
    )
    return windows.max(axis=(3, 4)).astype(np.float32)


def reorg(x, stride=2):
    """
    Space-to-depth (passthrough).
    x: [C, H, W]  →  [C*stride*stride, H//stride, W//stride]
    """
    C, H, W = x.shape
    S = stride
    OH, OW = H // S, W // S
    # x → [C, OH, S, OW, S] → [C, S, S, OH, OW] → [C*S*S, OH, OW]
    out = x.reshape(C, OH, S, OW, S)
    out = out.transpose(0, 2, 4, 1, 3)              # [C, S, S, OH, OW]
    return out.reshape(C * S * S, OH, OW).astype(np.float32)


# ---------------------------------------------------------------------------
# YOLOv2 full model
# ---------------------------------------------------------------------------

class YOLOv2:
    """
    Full 28-layer YOLOv2.

    Parameters
    ----------
    layers : list of dicts {'weights': np.ndarray, 'biases': np.ndarray}
             as returned by darknet_weights.load_weights() or make_synthetic_weights()
    quantised : bool
             If True, quantise inputs/weights/outputs to ap_fixed<16,8> at each step.
    """

    def __init__(self, layers: list, quantised: bool = False):
        self.layers    = layers
        self.quantised = quantised
        self.layer_outputs = {}   # keyed by layer index

    def _conv(self, layer_idx, conv_idx, x, relu=True):
        lw = self.layers[conv_idx]
        y = conv2d(x, lw['weights'], lw['biases'],
                   stride=1,
                   pad=(1 if lw['weights'].shape[2] == 3 else 0),
                   quantised=self.quantised)
        if relu:
            y = leaky_relu(y, self.quantised)
        self.layer_outputs[layer_idx] = y
        return y

    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        img : float32 [3, 416, 416], values in [0, 1]
        returns: float32 [125, 13, 13]  raw detection output
        """
        Q = self.quantised
        x = to_fixed(img) if Q else img.astype(np.float32)

        ci = 0   # conv-layer index into self.layers[]

        def conv(li, relu=True):
            nonlocal ci, x
            x = self._conv(li, ci, x, relu=relu)
            ci += 1

        def pool(li):
            nonlocal x
            x = maxpool(x)
            self.layer_outputs[li] = x

        # ----- Backbone -----
        conv(0);   pool(1)
        conv(2);   pool(3)
        conv(4);   conv(5);  conv(6);   pool(7)
        conv(8);   conv(9);  conv(10);  pool(11)
        conv(12);  conv(13); conv(14);  conv(15); conv(16)

        passthrough = x.copy()      # layer-16 output saved for passthrough
        self.layer_outputs['pt'] = passthrough

        pool(17)
        conv(18);  conv(19); conv(20);  conv(21); conv(22)
        conv(23);  conv(24)

        branch = x.copy()           # layer-24 output

        # Reorg + concat
        pt_reorg = reorg(passthrough, stride=2)   # 26×26×512 → 13×13×2048
        self.layer_outputs[25] = pt_reorg
        if Q:
            pt_reorg = to_fixed(pt_reorg)
            branch   = to_fixed(branch)
        x = np.concatenate([branch, pt_reorg], axis=0)   # 13×13×3072
        self.layer_outputs['concat'] = x

        conv(26)           # 3072 → 1024
        conv(27, relu=False)  # 1024 → 125, linear

        return x


# ---------------------------------------------------------------------------
# Decode detections
# ---------------------------------------------------------------------------

ANCHOR_W = [1.3221, 3.1923, 5.5556, 9.8028, 3.6518]
ANCHOR_H = [1.7314, 4.0094, 4.3655, 9.0094, 3.6833]
VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car',
    'cat','chair','cow','diningtable','dog','horse','motorbike',
    'person','pottedplant','sheep','sofa','train','tvmonitor'
]


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def decode_detections(out: np.ndarray, thresh=0.3, nms_thresh=0.45):
    """
    out : [125, 13, 13]
    Returns list of {'box': [cx,cy,w,h], 'conf': float, 'class': int, 'label': str}
    (cx,cy,w,h) are in 416-pixel units.
    """
    num_anchors  = 5
    num_classes  = 20
    grid_h, grid_w = 13, 13
    CELL = 416 / 13      # pixels per grid cell ≈ 32

    dets = []
    for a in range(num_anchors):
        for gy in range(grid_h):
            for gx in range(grid_w):
                base = a * (5 + num_classes)
                tx   = out[base + 0, gy, gx]
                ty   = out[base + 1, gy, gx]
                tw   = out[base + 2, gy, gx]
                th   = out[base + 3, gy, gx]
                tc   = out[base + 4, gy, gx]

                obj = _sigmoid(tc)
                if obj < thresh:
                    continue

                cls_scores = _softmax(out[base+5 : base+5+num_classes, gy, gx])
                best_cls   = int(np.argmax(cls_scores))
                confidence = obj * cls_scores[best_cls]
                if confidence < thresh:
                    continue

                bx = (_sigmoid(tx) + gx) / grid_w * 416
                by = (_sigmoid(ty) + gy) / grid_h * 416
                bw = ANCHOR_W[a] * np.exp(tw) / grid_w * 416
                bh = ANCHOR_H[a] * np.exp(th) / grid_h * 416

                dets.append({
                    'box':       [bx, by, bw, bh],
                    'conf':      float(confidence),
                    'obj':       float(obj),
                    'class':     best_cls,
                    'label':     VOC_CLASSES[best_cls],
                    'cls_score': float(cls_scores[best_cls]),
                })

    # NMS per class
    dets.sort(key=lambda d: d['conf'], reverse=True)
    keep = []
    for d in dets:
        suppress = False
        for k in keep:
            if k['class'] != d['class']:
                continue
            bx,by,bw,bh = d['box']; kx,ky,kw,kh = k['box']
            ix = max(0, min(bx+bw/2,kx+kw/2) - max(bx-bw/2,kx-kw/2))
            iy = max(0, min(by+bh/2,ky+kh/2) - max(by-bh/2,ky-kh/2))
            inter = ix * iy
            iou = inter / (bw*bh + kw*kh - inter + 1e-6)
            if iou > nms_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(d)
    return keep
