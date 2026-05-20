#include "yolov2_accel.h"
#include "conv_engine.h"
#include "pool_engine.h"
#include "reorg_engine.h"

/*
 * Top-level function.
 *
 * The AXI4 master (gmem) and AXI4-Lite slave (ctrl) interfaces are
 * declared through HLS pragmas.  The PS ARM core writes the configuration
 * registers through AXI-Lite, then sets cmd to trigger execution.
 * Vivado HLS generates a handshake / ap_done signal that the PS polls.
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
    unsigned int  flags)
{
    // AXI4 master: 64-bit data bus, 32-bit addressing
    #pragma HLS INTERFACE m_axi     port=gmem     bundle=gmem offset=slave \
        depth=16777216 max_read_burst_length=256 max_write_burst_length=256

    // AXI4-Lite slave: one bundle for all control registers + ap_return
    #pragma HLS INTERFACE s_axilite port=cmd      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_addr  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_addr bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=wt_addr  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_h     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_w     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=in_c     bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=out_c    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=ksize    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=stride   bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=pad      bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=flags    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return   bundle=ctrl

    bool use_lrelu = (flags & FLAG_LRELU) != 0;

    switch (cmd) {
        case CMD_CONV:
            conv_engine(gmem,
                        in_addr, out_addr, wt_addr,
                        in_h, in_w, in_c, out_c,
                        ksize, stride, pad, use_lrelu);
            break;

        case CMD_POOL:
            pool_engine(gmem, in_addr, out_addr, in_h, in_w, in_c, stride);
            break;

        case CMD_REORG:
            reorg_engine(gmem, in_addr, out_addr, in_h, in_w, in_c, stride);
            break;

        default:
            break;
    }
}
