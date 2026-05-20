#pragma once

#include "defines.h"

/*
 * Tiled convolution engine (3×3 or 1×1), bias add, optional leaky-ReLU.
 *
 * Algorithm
 *   Outer loops: spatial tiles (TILE_H × TILE_W) × output-channel strips (CSTRIP_O)
 *   Inner loops: input-channel strips (CSTRIP_I) × kernel positions
 *   Accumulates partial sums in a local BRAM array; writes full output tile
 *   to DDR via burst transfer.
 *
 *   For a CSTRIP_O=8 output strip with the inner IC/kh/kw loop pipelined at II=1
 *   and the 8 output accumulators unrolled, the design uses 8 DSP slices.
 *
 * Weight/bias layout in DDR  (all data_t, 16-bit):
 *   weights[out_c][in_c][ksize][ksize]   (row-major)
 *   biases[out_c]
 */
void conv_engine(
    axi_word_t   *gmem,
    unsigned int  in_addr,
    unsigned int  out_addr,
    unsigned int  wt_addr,
    unsigned int  in_h,
    unsigned int  in_w,
    unsigned int  in_c,
    unsigned int  out_c,
    unsigned int  ksize,
    unsigned int  stride,
    unsigned int  pad,
    bool          use_lrelu
);
