"""
Fixed-point arithmetic utilities matching the HLS ap_fixed<16,8> type.

ap_fixed<16,8>:
  - 16 total bits
  - 8 integer bits (including sign)  → range [-128, 127.996...)
  - 8 fractional bits                → resolution 1/256 ≈ 0.0039
  - Rounding: round-to-nearest
  - Overflow: saturate
"""

import numpy as np

INT_BITS  = 8
FRAC_BITS = 8
TOTAL_BITS = INT_BITS + FRAC_BITS   # 16

SCALE   = float(1 << FRAC_BITS)     # 256.0
MAX_VAL =  (1 << (TOTAL_BITS - 1)) - 1   # 32767  → 127.996...
MIN_VAL = -(1 << (TOTAL_BITS - 1))       # -32768 → -128.0


def to_fixed(x: np.ndarray) -> np.ndarray:
    """
    Quantise float32 array to ap_fixed<16,8> values returned as float32.
    Operation: round(x * 256) / 256  with saturation to [-128, 127.996)
    """
    x = np.asarray(x, dtype=np.float64)
    raw = np.round(x * SCALE)                    # to integer counts
    raw = np.clip(raw, MIN_VAL, MAX_VAL)         # saturate
    return (raw / SCALE).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    """
    Encode fixed-point values as int16 (the bit pattern stored in DDR /
    written to HLS test vectors).
    """
    x = np.asarray(x, dtype=np.float64)
    raw = np.round(x * SCALE)
    raw = np.clip(raw, MIN_VAL, MAX_VAL)
    return raw.astype(np.int16)


def from_int16(raw: np.ndarray) -> np.ndarray:
    """Decode int16 bit pattern back to float32 fixed-point value."""
    return (raw.astype(np.float32)) / SCALE


def quantize_weights(w: np.ndarray) -> np.ndarray:
    """Quantise float32 weights to ap_fixed<16,8> (same type)."""
    return to_fixed(w)


def fixed_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Simulate ap_fixed<32,16> accumulator product then round back.
    In hardware:  acc_t (32,16) = data_t (16,8) * wt_t (16,8)
    Result range: [-128*128, 128*128) = [-16384, 16384)
    """
    return to_fixed(a * b)   # simplified: quantise the product


def write_bin(path: str, arr: np.ndarray) -> None:
    """Write array as little-endian int16 binary file."""
    to_int16(arr).flatten().tofile(path)


def read_bin(path: str, shape=None) -> np.ndarray:
    """Read little-endian int16 binary file and return float32 array."""
    raw = np.fromfile(path, dtype='<i2')
    arr = from_int16(raw)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-noise ratio in dB between reference and test arrays."""
    signal = np.mean(ref ** 2)
    noise  = np.mean((ref - test) ** 2)
    if noise < 1e-12:
        return float('inf')
    return 10.0 * np.log10(signal / noise)


def layer_stats(ref: np.ndarray, test: np.ndarray, name: str = "") -> dict:
    """Compute comparison statistics for a layer output."""
    diff = np.abs(ref.astype(np.float32) - test.astype(np.float32))
    stats = {
        "layer":    name,
        "mae":      float(np.mean(diff)),
        "max_err":  float(np.max(diff)),
        "snr_db":   snr_db(ref.astype(np.float32), test.astype(np.float32)),
        "pct_ok":   float(100.0 * np.mean(diff <= 1.0 / SCALE)),  # within 1 LSB
    }
    return stats
