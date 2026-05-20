#include "pool_engine.h"

static inline ap_uint<16> pack_d(data_t v) {
    ap_uint<16> r; r.range(15,0) = v.range(15,0); return r;
}
static inline data_t unpack_d(ap_uint<16> r) {
    data_t v; v.range(15,0) = r.range(15,0); return v;
}

// Read one data_t from DDR
static data_t read_val(axi_word_t *gmem, unsigned int byte_off) {
    unsigned int wo = byte_off >> 3;
    int lane = (byte_off & 7) >> 1;
    return unpack_d(gmem[wo].range(16*lane+15, 16*lane));
}

// Write one data_t to DDR (read-modify-write)
static void write_val(axi_word_t *gmem, unsigned int byte_off, data_t v) {
    unsigned int wo = byte_off >> 3;
    int lane = (byte_off & 7) >> 1;
    axi_word_t w = gmem[wo];
    w.range(16*lane+15, 16*lane) = pack_d(v);
    gmem[wo] = w;
}

void pool_engine(
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

    for (unsigned int ic = 0; ic < in_c; ic++) {
        for (unsigned int oh = 0; oh < out_h; oh++) {
            for (unsigned int ow = 0; ow < out_w; ow++) {
                #pragma HLS PIPELINE II=4
                data_t mx = data_t(-128);
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        unsigned ih = oh * stride + kh;
                        unsigned iw = ow * stride + kw;
                        if (ih < in_h && iw < in_w) {
                            unsigned src = in_addr
                                + (ic * in_h * in_w + ih * in_w + iw) * 2;
                            data_t v = read_val(gmem, src);
                            if (v > mx) mx = v;
                        }
                    }
                }
                unsigned dst = out_addr
                    + (ic * out_h * out_w + oh * out_w + ow) * 2;
                write_val(gmem, dst, mx);
            }
        }
    }
}
