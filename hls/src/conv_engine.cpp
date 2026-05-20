#include "conv_engine.h"

// ---------------------------------------------------------------------------
// Helpers: pack / unpack 16-bit fixed-point ↔ AXI-64 words
// ---------------------------------------------------------------------------

static inline ap_uint<16> pack_d(data_t v) {
    ap_uint<16> r;
    r.range(15, 0) = v.range(15, 0);
    return r;
}
static inline data_t unpack_d(ap_uint<16> r) {
    data_t v;
    v.range(15, 0) = r.range(15, 0);
    return v;
}
static inline wt_t unpack_w(ap_uint<16> r) {
    wt_t v;
    v.range(15, 0) = r.range(15, 0);
    return v;
}

// ---------------------------------------------------------------------------
// AXI burst helpers (sub-word granularity through packing)
// ---------------------------------------------------------------------------

static void ddr_to_data(axi_word_t *gmem, unsigned int byte_off,
                         data_t *buf, unsigned int n) {
    unsigned int w0 = byte_off >> 3;
    unsigned int nw = (n + VALS_PER_WORD - 1) / VALS_PER_WORD;
    for (unsigned int wi = 0; wi < nw; wi++) {
        #pragma HLS PIPELINE II=1
        axi_word_t word = gmem[w0 + wi];
        for (int j = 0; j < VALS_PER_WORD; j++) {
            unsigned idx = wi * VALS_PER_WORD + j;
            if (idx < n)
                buf[idx] = unpack_d(word.range(16*j+15, 16*j));
        }
    }
}

static void ddr_to_wt(axi_word_t *gmem, unsigned int byte_off,
                       wt_t *buf, unsigned int n) {
    unsigned int w0 = byte_off >> 3;
    unsigned int nw = (n + VALS_PER_WORD - 1) / VALS_PER_WORD;
    for (unsigned int wi = 0; wi < nw; wi++) {
        #pragma HLS PIPELINE II=1
        axi_word_t word = gmem[w0 + wi];
        for (int j = 0; j < VALS_PER_WORD; j++) {
            unsigned idx = wi * VALS_PER_WORD + j;
            if (idx < n)
                buf[idx] = unpack_w(word.range(16*j+15, 16*j));
        }
    }
}

static void data_to_ddr(axi_word_t *gmem, unsigned int byte_off,
                         const data_t *buf, unsigned int n) {
    unsigned int w0 = byte_off >> 3;
    unsigned int nw = (n + VALS_PER_WORD - 1) / VALS_PER_WORD;
    for (unsigned int wi = 0; wi < nw; wi++) {
        #pragma HLS PIPELINE II=1
        axi_word_t word = 0;
        for (int j = 0; j < VALS_PER_WORD; j++) {
            unsigned idx = wi * VALS_PER_WORD + j;
            if (idx < n)
                word.range(16*j+15, 16*j) = pack_d(buf[idx]);
        }
        gmem[w0 + wi] = word;
    }
}

// ---------------------------------------------------------------------------
// Tile-strip convolution kernel
//
//   Processes one spatial tile (TILE_H × TILE_W) with CSTRIP_O output
//   channels and CSTRIP_I input channels at a time.
//   The inner IC × KH × KW loop is pipelined (II=1); the CSTRIP_O output
//   accumulators are fully unrolled → 8 DSPs in parallel.
// ---------------------------------------------------------------------------

static void conv_tile_strip(
    const data_t in_tile[CSTRIP_I][TILE_H+2][TILE_W+2],
    const wt_t   wt_buf [CSTRIP_O][CSTRIP_I][3][3],
    acc_t        acc    [CSTRIP_O][TILE_H][TILE_W],
    int ic_local, int oc_local,
    int ksize, int tile_oh, int tile_ow,
    bool accumulate)
{
    #pragma HLS ARRAY_PARTITION variable=in_tile complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wt_buf  complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wt_buf  complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wt_buf  complete dim=4
    #pragma HLS ARRAY_PARTITION variable=acc     complete dim=1

    for (int oh = 0; oh < tile_oh; oh++) {
        for (int ow = 0; ow < tile_ow; ow++) {
            for (int ic = 0; ic < ic_local; ic++) {
                for (int kh = 0; kh < ksize; kh++) {
                    for (int kw = 0; kw < ksize; kw++) {
                        #pragma HLS PIPELINE II=1
                        data_t in_val = in_tile[ic][oh+kh][ow+kw];
                        for (int oc = 0; oc < CSTRIP_O; oc++) {
                            #pragma HLS UNROLL
                            acc_t term = in_val * wt_buf[oc][ic][kh][kw];
                            if (!accumulate && ic == 0 && kh == 0 && kw == 0)
                                acc[oc][oh][ow] = term;
                            else
                                acc[oc][oh][ow] += term;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// conv_engine – top-level tiled convolution
// ---------------------------------------------------------------------------

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
    bool          use_lrelu)
{
    unsigned int out_h = (in_h + 2*pad - ksize) / stride + 1;
    unsigned int out_w = (in_w + 2*pad - ksize) / stride + 1;

    // Bias buffer: biases sit right after weights in DDR
    unsigned int wt_elem  = out_c * in_c * ksize * ksize;
    unsigned int b_offset = wt_addr + wt_elem * 2;   // byte offset (2 B per elem)

    wt_t biases[1024];
    #pragma HLS ARRAY_PARTITION variable=biases cyclic factor=8

    // Load all biases once
    ddr_to_wt(gmem, b_offset, biases, out_c);

    // Local weight strip buffer: [CSTRIP_O][CSTRIP_I][3][3]
    wt_t wt_buf[CSTRIP_O][CSTRIP_I][3][3];
    #pragma HLS ARRAY_PARTITION variable=wt_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wt_buf complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wt_buf complete dim=4

    // Input tile buffer:  [CSTRIP_I][TILE_H+2][TILE_W+2]
    data_t in_tile[CSTRIP_I][TILE_H+2][TILE_W+2];
    #pragma HLS ARRAY_PARTITION variable=in_tile complete dim=1

    // Accumulator: [CSTRIP_O][TILE_H][TILE_W]
    acc_t acc[CSTRIP_O][TILE_H][TILE_W];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    // Output tile: flattened strip  [CSTRIP_O × TILE_H × TILE_W]
    data_t out_tile[CSTRIP_O * TILE_H * TILE_W];

    // Spatial tiling
    for (unsigned int oh_tile = 0; oh_tile < out_h; oh_tile += TILE_H) {
        unsigned int tile_oh = (oh_tile + TILE_H <= out_h) ? TILE_H : (out_h - oh_tile);

        for (unsigned int ow_tile = 0; ow_tile < out_w; ow_tile += TILE_W) {
            unsigned int tile_ow = (ow_tile + TILE_W <= out_w) ? TILE_W : (out_w - ow_tile);

            // Output channel strip
            for (unsigned int oc_s = 0; oc_s < out_c; oc_s += CSTRIP_O) {
                unsigned int oc_local = ((oc_s + CSTRIP_O) <= out_c) ? CSTRIP_O : (out_c - oc_s);

                // Clear accumulator
                for (int oc = 0; oc < CSTRIP_O; oc++)
                    for (int oh = 0; oh < TILE_H; oh++)
                        for (int ow = 0; ow < TILE_W; ow++) {
                            #pragma HLS PIPELINE II=1
                            acc[oc][oh][ow] = acc_t(0);
                        }

                // Input channel strip
                for (unsigned int ic_s = 0; ic_s < in_c; ic_s += CSTRIP_I) {
                    unsigned int ic_local = ((ic_s + CSTRIP_I) <= in_c) ? CSTRIP_I : (in_c - ic_s);

                    // ---- Load input tile from DDR --------------------------------
                    for (int ic = 0; ic < (int)ic_local; ic++) {
                        unsigned int ic_abs = ic_s + ic;
                        for (int r = 0; r < (int)(tile_oh + ksize - 1); r++) {
                            int ih = (int)oh_tile * (int)stride - (int)pad + r;
                            for (int c = 0; c < (int)(tile_ow + ksize - 1); c++) {
                                int iw = (int)ow_tile * (int)stride - (int)pad + c;
                                data_t val = data_t(0);
                                if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                                    unsigned int src_off = in_addr
                                        + (ic_abs * in_h * in_w + ih * in_w + iw) * 2;
                                    axi_word_t w = gmem[src_off >> 3];
                                    int lane = (src_off & 7) >> 1;  // which 16-bit slot
                                    val = unpack_d(w.range(16*lane+15, 16*lane));
                                }
                                in_tile[ic][r][c] = val;
                            }
                        }
                    }

                    // ---- Load weight strip from DDR ------------------------------
                    for (int oc = 0; oc < (int)oc_local; oc++) {
                        unsigned int oc_abs = oc_s + oc;
                        for (int ic = 0; ic < (int)ic_local; ic++) {
                            unsigned int ic_abs = ic_s + ic;
                            unsigned int wt_off = wt_addr
                                + (oc_abs * in_c * ksize * ksize
                                 + ic_abs * ksize * ksize) * 2;
                            wt_t tmp[9];
                            ddr_to_wt(gmem, wt_off, tmp, ksize * ksize);
                            for (int k = 0; k < ksize*ksize; k++) {
                                #pragma HLS PIPELINE II=1
                                wt_buf[oc][ic][k/ksize][k%ksize] = tmp[k];
                            }
                        }
                    }

                    // ---- Accumulate partial convolution --------------------------
                    conv_tile_strip(in_tile, wt_buf, acc,
                                    ic_local, oc_local, ksize,
                                    tile_oh, tile_ow,
                                    /*accumulate=*/ (ic_s > 0));
                }

                // ---- Apply bias + activation, store tile to DDR -----------------
                for (int oc = 0; oc < (int)oc_local; oc++) {
                    unsigned int oc_abs = oc_s + oc;
                    for (int oh = 0; oh < (int)tile_oh; oh++) {
                        for (int ow = 0; ow < (int)tile_ow; ow++) {
                            #pragma HLS PIPELINE II=1
                            data_t result = (data_t)(acc[oc][oh][ow] + biases[oc_abs]);
                            if (use_lrelu && result < data_t(0))
                                result = (data_t)(LEAKY_SLOPE * result);

                            unsigned int dst_oh = oh_tile + oh;
                            unsigned int dst_ow = ow_tile + ow;
                            unsigned int dst_off = out_addr
                                + (oc_abs * out_h * out_w + dst_oh * out_w + dst_ow) * 2;
                            // Single 16-bit write via read-modify-write on 64-bit word
                            unsigned int word_off = dst_off >> 3;
                            int lane = (dst_off & 7) >> 1;
                            axi_word_t wrd = gmem[word_off];
                            wrd.range(16*lane+15, 16*lane) = pack_d(result);
                            gmem[word_off] = wrd;
                        }
                    }
                }
            }
        }
    }
}
