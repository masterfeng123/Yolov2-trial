#pragma once

#include "defines.h"

/*
 * Top-level accelerator function.
 *
 * Interfaces
 *   gmem  – AXI4 master  (64-bit) → PS DDR burst read/write
 *   ctrl  – AXI4-Lite slave       → register map below
 *   ap_return is exposed as a status/interrupt register
 *
 * AXI-Lite register map (Vivado HLS auto-assigns offsets):
 *   cmd      – write CMD_CONV / CMD_POOL / CMD_REORG to launch
 *   in_addr  – byte address of input  feature map in DDR
 *   out_addr – byte address of output feature map in DDR
 *   wt_addr  – byte address of weight blob in DDR (CONV only)
 *   in_h,in_w,in_c  – input spatial dimensions and channels
 *   out_c    – output channels (CONV only; POOL/REORG inherit in_c)
 *   ksize    – kernel size: 1 or 3 (CONV only)
 *   stride   – conv/pool stride
 *   pad      – zero-pad amount (0 or 1)
 *   flags    – bit0 = apply_leaky_relu
 *
 * Weight blob layout in DDR (folded, 16-bit each):
 *   [out_c × in_c × ksize × ksize]  weights
 *   [out_c]                          biases
 */
void yolov2_accel(
    axi_word_t   *gmem,
    unsigned int  cmd,
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
    unsigned int  flags
);
