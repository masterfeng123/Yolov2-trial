#include "reorg_engine.h"

static inline ap_uint<16> pack_d(data_t v) {
    ap_uint<16> r; r.range(15,0) = v.range(15,0); return r;
}
static inline data_t unpack_d(ap_uint<16> r) {
    data_t v; v.range(15,0) = r.range(15,0); return v;
}

static data_t read_val(axi_word_t *gmem, unsigned int byte_off) {
    unsigned wo = byte_off >> 3;
    int lane = (byte_off & 7) >> 1;
    return unpack_d(gmem[wo].range(16*lane+15, 16*lane));
}

static void write_val(axi_word_t *gmem, unsigned int byte_off, data_t v) {
    unsigned wo = byte_off >> 3;
    int lane = (byte_off & 7) >> 1;
    axi_word_t w = gmem[wo];
    w.range(16*lane+15, 16*lane) = pack_d(v);
    gmem[wo] = w;
}

void reorg_engine(
    axi_word_t   *gmem,
    unsigned int  in_addr,
    unsigned int  out_addr,
    unsigned int  in_h,
    unsigned int  in_w,
    unsigned int  in_c,
    unsigned int  stride)
{
    unsigned int out_h = in_h / stride;
    unsigned int out_w = in_w / stride;
    unsigned int out_c = in_c * stride * stride;   // 512*4 = 2048

    for (unsigned int ic = 0; ic < in_c; ic++) {
        for (unsigned int kh = 0; kh < stride; kh++) {
            for (unsigned int kw = 0; kw < stride; kw++) {
                unsigned int oc = ic * stride * stride + kh * stride + kw;
                for (unsigned int oh = 0; oh < out_h; oh++) {
                    for (unsigned int ow = 0; ow < out_w; ow++) {
                        #pragma HLS PIPELINE II=1
                        unsigned int ih = oh * stride + kh;
                        unsigned int iw = ow * stride + kw;
                        unsigned int src = in_addr
                            + (ic * in_h * in_w + ih * in_w + iw) * 2;
                        unsigned int dst = out_addr
                            + (oc * out_h * out_w + oh * out_w + ow) * 2;
                        write_val(gmem, dst, read_val(gmem, src));
                    }
                }
            }
        }
    }
}
