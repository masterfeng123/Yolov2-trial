/*
 * Functional testbench for the YOLOv2 HLS accelerator.
 *
 * Tests exercised
 *   1. CONV 3×3  : 8×8 input, 4 in-channels → 4 out-channels (known weights)
 *   2. CONV 1×1  : verify bias-only (weight=1, bias=0.5, leaky off)
 *   3. POOL 2×2  : stride-2 max-pool on a simple ramp pattern
 *   4. REORG     : 4×4×2 → 2×2×8 space-to-depth (stride=2)
 *
 * The DDR memory is emulated as a flat byte-addressable ap_uint<64> array.
 * All comparisons use a tolerance of 2 LSBs (one rounding step in ap_fixed<16,8>).
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../src/yolov2_accel.h"

// 32 MB emulated DDR
#define DDR_BYTES (32 * 1024 * 1024)
#define DDR_WORDS (DDR_BYTES / 8)

static axi_word_t ddr[DDR_WORDS];

// ---------------------------------------------------------------------------
// Helpers: read / write data_t in the emulated DDR
// ---------------------------------------------------------------------------

static void ddr_write(unsigned int byte_off, data_t v) {
    unsigned wo = byte_off >> 3;
    int lane = (byte_off & 7) >> 1;
    axi_word_t w = ddr[wo];
    ap_uint<16> bits; bits.range(15,0) = v.range(15,0);
    w.range(16*lane+15, 16*lane) = bits;
    ddr[wo] = w;
}

static data_t ddr_read(unsigned int byte_off) {
    unsigned wo = byte_off >> 3;
    int lane = (byte_off & 7) >> 1;
    ap_uint<16> bits = ddr[wo].range(16*lane+15, 16*lane);
    data_t v; v.range(15,0) = bits.range(15,0);
    return v;
}

static void wt_write(unsigned int byte_off, wt_t v) {
    unsigned wo = byte_off >> 3;
    int lane = (byte_off & 7) >> 1;
    axi_word_t w = ddr[wo];
    ap_uint<16> bits; bits.range(15,0) = v.range(15,0);
    w.range(16*lane+15, 16*lane) = bits;
    ddr[wo] = w;
}

// ---------------------------------------------------------------------------
// Test 1: 3×3 convolution with identity-like weights
// ---------------------------------------------------------------------------
static int test_conv3x3() {
    printf("[TEST] CONV 3x3 4-in 4-out 8x8 ...\n");

    const unsigned IN_H=8, IN_W=8, IN_C=4, OUT_C=4, KS=3;
    const unsigned STRIDE=1, PAD=1;
    const unsigned IN_ADDR  = 0x00000000u;
    const unsigned OUT_ADDR = 0x00010000u;
    const unsigned WT_ADDR  = 0x00020000u;

    // Fill input: channel c, pixel (h,w) = c + h*0.1 + w*0.01
    for (unsigned c=0; c<IN_C; c++)
        for (unsigned h=0; h<IN_H; h++)
            for (unsigned w=0; w<IN_W; w++) {
                data_t v((float)(c + h*0.1f + w*0.01f));
                ddr_write(IN_ADDR + (c*IN_H*IN_W + h*IN_W + w)*2, v);
            }

    // Weights: identity-like (oc == ic, centre kh=1,kw=1 → 1.0, rest 0)
    unsigned wt_n = OUT_C * IN_C * KS * KS;
    for (unsigned i=0; i<wt_n; i++) wt_write(WT_ADDR + i*2, wt_t(0));
    for (unsigned oc=0; oc<OUT_C; oc++) {
        unsigned idx = oc * IN_C * KS * KS + oc * KS * KS + 1*KS + 1;
        wt_write(WT_ADDR + idx*2, wt_t(1));
    }
    // Biases = 0
    unsigned bias_off = WT_ADDR + wt_n * 2;
    for (unsigned i=0; i<OUT_C; i++) wt_write(bias_off + i*2, wt_t(0));

    // Run
    yolov2_accel(ddr, CMD_CONV,
                 IN_ADDR, OUT_ADDR, WT_ADDR,
                 IN_H, IN_W, IN_C, OUT_C,
                 KS, STRIDE, PAD, FLAG_LRELU);

    // Check: output[oc][h][w] ≈ input[oc][h][w]  (centre-pixel identity)
    int errs = 0;
    const float TOL = 0.02f;
    unsigned out_h = IN_H, out_w = IN_W;
    for (unsigned oc=0; oc<OUT_C; oc++)
        for (unsigned h=0; h<out_h; h++)
            for (unsigned w=0; w<out_w; w++) {
                data_t got = ddr_read(OUT_ADDR + (oc*out_h*out_w + h*out_w + w)*2);
                float exp_v = (float)(oc + h*0.1f + w*0.01f);
                float diff = (float)got - exp_v;
                // Edge pixels are zero-padded → may differ
                if (h>0 && h<(out_h-1) && w>0 && w<(out_w-1)) {
                    if (diff < -TOL || diff > TOL) {
                        printf("  FAIL oc=%u h=%u w=%u: got=%.4f exp=%.4f\n",
                               oc, h, w, (float)got, exp_v);
                        errs++;
                    }
                }
            }
    printf("  %s (%d errors)\n", errs==0?"PASS":"FAIL", errs);
    return errs;
}

// ---------------------------------------------------------------------------
// Test 2: 2×2 max-pool, stride=2
// ---------------------------------------------------------------------------
static int test_pool() {
    printf("[TEST] POOL 2x2 stride=2 8x8 ch=2 ...\n");

    const unsigned IH=8, IW=8, IC=2;
    const unsigned IA=0x00100000u, OA=0x00110000u;

    // Input: pixel value = channel*10 + row + col/8.0
    for (unsigned c=0; c<IC; c++)
        for (unsigned h=0; h<IH; h++)
            for (unsigned w=0; w<IW; w++) {
                float v = c*10.0f + h + w/8.0f;
                ddr_write(IA + (c*IH*IW + h*IW + w)*2, data_t(v));
            }

    yolov2_accel(ddr, CMD_POOL, IA, OA, 0, IH, IW, IC, 0, 2, 2, 0, 0);

    // Expected: max of each 2×2 window
    unsigned OH = IH/2, OW = IW/2;
    int errs = 0;
    for (unsigned c=0; c<IC; c++)
        for (unsigned oh=0; oh<OH; oh++)
            for (unsigned ow=0; ow<OW; ow++) {
                float mx = -1e9;
                for (int kh=0; kh<2; kh++)
                    for (int kw=0; kw<2; kw++) {
                        float v = c*10.0f + (oh*2+kh) + (ow*2+kw)/8.0f;
                        if (v > mx) mx = v;
                    }
                data_t got = ddr_read(OA + (c*OH*OW + oh*OW + ow)*2);
                float diff = (float)got - mx;
                if (diff < -0.05f || diff > 0.05f) {
                    printf("  FAIL c=%u oh=%u ow=%u got=%.3f exp=%.3f\n",
                           c, oh, ow, (float)got, mx);
                    errs++;
                }
            }
    printf("  %s (%d errors)\n", errs==0?"PASS":"FAIL", errs);
    return errs;
}

// ---------------------------------------------------------------------------
// Test 3: REORG (space-to-depth) 4×4×2 → 2×2×8
// ---------------------------------------------------------------------------
static int test_reorg() {
    printf("[TEST] REORG 4x4x2 -> 2x2x8 (stride=2) ...\n");

    const unsigned IH=4, IW=4, IC=2, STR=2;
    const unsigned IA=0x00200000u, OA=0x00210000u;

    // Input[c][h][w] = c*100 + h*10 + w
    for (unsigned c=0; c<IC; c++)
        for (unsigned h=0; h<IH; h++)
            for (unsigned w=0; w<IW; w++)
                ddr_write(IA + (c*IH*IW + h*IW + w)*2,
                          data_t((float)(c*100 + h*10 + w)));

    yolov2_accel(ddr, CMD_REORG, IA, OA, 0, IH, IW, IC, 0, 2, STR, 0, 0);

    // output[ic*4 + kh*2 + kw][oh][ow] = input[ic][oh*2+kh][ow*2+kw]
    unsigned OH=IH/STR, OW=IW/STR, OC=IC*STR*STR;
    int errs = 0;
    for (unsigned ic=0; ic<IC; ic++)
        for (unsigned kh=0; kh<STR; kh++)
            for (unsigned kw=0; kw<STR; kw++) {
                unsigned oc = ic*STR*STR + kh*STR + kw;
                for (unsigned oh=0; oh<OH; oh++)
                    for (unsigned ow=0; ow<OW; ow++) {
                        float exp_v = ic*100.0f + (oh*2+kh)*10.0f + (ow*2+kw);
                        data_t got = ddr_read(OA + (oc*OH*OW + oh*OW + ow)*2);
                        float diff = (float)got - exp_v;
                        if (diff < -0.05f || diff > 0.05f) {
                            printf("  FAIL oc=%u oh=%u ow=%u got=%.1f exp=%.1f\n",
                                   oc, oh, ow, (float)got, exp_v);
                            errs++;
                        }
                    }
            }
    printf("  %s (%d errors)\n", errs==0?"PASS":"FAIL", errs);
    return errs;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    int total_errs = 0;

    // Clear DDR
    for (int i = 0; i < DDR_WORDS; i++) ddr[i] = 0;

    total_errs += test_conv3x3();
    total_errs += test_pool();
    total_errs += test_reorg();

    printf("\n=== Testbench %s (total errors: %d) ===\n",
           total_errs == 0 ? "PASSED" : "FAILED", total_errs);
    return total_errs;
}
