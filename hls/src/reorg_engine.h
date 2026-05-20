#pragma once

#include "defines.h"

/*
 * Space-to-depth (reorg / passthrough) layer.
 *
 * YOLOv2 uses stride=2 to convert the 26×26×512 feature map from
 * layer-16 into 13×13×2048, which is then concatenated with the
 * 13×13×1024 output of layer-24 to produce the 13×13×3072 input
 * of the final detection convolution.
 *
 * Output layout: output[c*stride*stride + (kh*stride+kw)][oh][ow]
 *              = input[c][oh*stride+kh][ow*stride+kw]
 */
void reorg_engine(
    axi_word_t   *gmem,
    unsigned int  in_addr,
    unsigned int  out_addr,
    unsigned int  in_h,
    unsigned int  in_w,
    unsigned int  in_c,
    unsigned int  stride
);
