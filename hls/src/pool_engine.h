#pragma once

#include "defines.h"

/*
 * Max-pooling engine.
 * Supports 2×2 kernel (the only size used in YOLOv2) with configurable stride.
 * Channels are processed independently.
 */
void pool_engine(
    axi_word_t   *gmem,
    unsigned int  in_addr,
    unsigned int  out_addr,
    unsigned int  in_h,
    unsigned int  in_w,
    unsigned int  in_c,
    unsigned int  stride
);
