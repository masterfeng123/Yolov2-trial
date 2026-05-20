#pragma once

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

/*
 * YOLOv2 HLS Accelerator – ZedBoard (XC7Z020)
 *
 * Quantisation scheme
 *   data_t  : ap_fixed<16,8>  – activations  (range ±128, step 0.0039)
 *   wt_t    : ap_fixed<16,8>  – weights / BN-folded biases (same range)
 *   acc_t   : ap_fixed<32,16> – accumulator  (range ±32768)
 *
 * Batch-normalisation is folded into conv weights offline:
 *   w'  = w  * gamma / sqrt(var + eps)
 *   b'  = beta - gamma * mean / sqrt(var + eps)
 * The hardware only performs:  y = sum(w' * x) + b'  [+ leaky-ReLU]
 */

// ---------------------------------------------------------------------------
// Arithmetic types
// ---------------------------------------------------------------------------
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  data_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  wt_t;
typedef ap_fixed<32,16, AP_RND, AP_SAT>  acc_t;

// AXI-64 bus: four 16-bit values per beat
static const int AXI_W         = 64;
static const int VALS_PER_WORD = AXI_W / 16;   // 4
typedef ap_uint<AXI_W> axi_word_t;

// ---------------------------------------------------------------------------
// Network topology (VOC 20-class YOLOv2, 416×416 input)
// ---------------------------------------------------------------------------
#define NET_IN_H        416
#define NET_IN_W        416
#define NET_IN_C        3
#define NET_NUM_CLASSES 20
#define NET_NUM_ANCHORS 5
#define NET_GRID_H      13
#define NET_GRID_W      13
#define NET_OUT_FILT    (NET_NUM_ANCHORS * (5 + NET_NUM_CLASSES))  // 125

static const wt_t LEAKY_SLOPE(0.1);

// ---------------------------------------------------------------------------
// Tiling parameters – chosen to fit in ZedBoard BRAM (4.9 Mb)
//
//   Weight buffer  : CSTRIP_O × CSTRIP_I × 3 × 3 × 2 B =  8×8×9×2 =  1152 B
//   Input tile     : CSTRIP_I × (TILE_H+2) × (TILE_W+2) × 2 B = 8×18×18×2 = 5184 B
//   Accumulator    : CSTRIP_O × TILE_H × TILE_W × 4 B = 8×16×16×4 = 8192 B
//   Total on-chip  : ~14 KB per accelerator invocation (well within limits)
// ---------------------------------------------------------------------------
#define TILE_H    16
#define TILE_W    16
#define CSTRIP_I   8   // input-channel strip
#define CSTRIP_O   8   // output-channel strip

// ---------------------------------------------------------------------------
// Accelerator commands (written to AXI-Lite CMD register)
// ---------------------------------------------------------------------------
#define CMD_IDLE   0x00u
#define CMD_CONV   0x01u
#define CMD_POOL   0x02u
#define CMD_REORG  0x03u

// flags field bit masks
#define FLAG_LRELU  0x01u   // apply leaky-ReLU after conv
