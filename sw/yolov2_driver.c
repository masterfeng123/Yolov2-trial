/*
 * YOLOv2 PS-side driver
 *
 * Targets Zynq Linux with /dev/mem access (or Xilinx xil_io on bare-metal).
 * For bare-metal replace the mmap section with direct pointer assignments.
 */

#include "yolov2_driver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

/* ============================================================
 * Layer configuration table
 * ============================================================
 * Each row: { cmd, in_h, in_w, in_c, out_c, ksize, stride, pad, flags }
 * POOL / REORG set out_c = in_c (unchanged); ksize unused for POOL/REORG.
 *
 * Layer 25 is a virtual CONCAT handled by DDR address arithmetic (no HW cmd).
 * The BN-fold step is done offline; all CONV layers just use FLAG_LRELU
 * except layer 27 (last conv, linear output).
 */
typedef struct {
    uint32_t cmd;
    uint32_t in_h, in_w, in_c;
    uint32_t out_c;
    uint32_t ksize, stride, pad, flags;
} LayerCfg;

#define F_LR  FLAG_LRELU
static const LayerCfg LAYERS[] = {
/*  0 */ { CMD_CONV, 416,416,  3,   32, 3,1,1, F_LR },
/*  1 */ { CMD_POOL, 416,416, 32,   32, 2,2,0, 0    },
/*  2 */ { CMD_CONV, 208,208, 32,   64, 3,1,1, F_LR },
/*  3 */ { CMD_POOL, 208,208, 64,   64, 2,2,0, 0    },
/*  4 */ { CMD_CONV, 104,104, 64,  128, 3,1,1, F_LR },
/*  5 */ { CMD_CONV, 104,104,128,   64, 1,1,0, F_LR },
/*  6 */ { CMD_CONV, 104,104, 64,  128, 3,1,1, F_LR },
/*  7 */ { CMD_POOL, 104,104,128,  128, 2,2,0, 0    },
/*  8 */ { CMD_CONV,  52, 52,128,  256, 3,1,1, F_LR },
/*  9 */ { CMD_CONV,  52, 52,256,  128, 1,1,0, F_LR },
/* 10 */ { CMD_CONV,  52, 52,128,  256, 3,1,1, F_LR },
/* 11 */ { CMD_POOL,  52, 52,256,  256, 2,2,0, 0    },
/* 12 */ { CMD_CONV,  26, 26,256,  512, 3,1,1, F_LR },
/* 13 */ { CMD_CONV,  26, 26,512,  256, 1,1,0, F_LR },
/* 14 */ { CMD_CONV,  26, 26,256,  512, 3,1,1, F_LR },
/* 15 */ { CMD_CONV,  26, 26,512,  256, 1,1,0, F_LR },
/* 16 */ { CMD_CONV,  26, 26,256,  512, 3,1,1, F_LR }, /* save → passthrough */
/* 17 */ { CMD_POOL,  26, 26,512,  512, 2,2,0, 0    },
/* 18 */ { CMD_CONV,  13, 13,512, 1024, 3,1,1, F_LR },
/* 19 */ { CMD_CONV,  13, 13,1024, 512, 1,1,0, F_LR },
/* 20 */ { CMD_CONV,  13, 13,512, 1024, 3,1,1, F_LR },
/* 21 */ { CMD_CONV,  13, 13,1024, 512, 1,1,0, F_LR },
/* 22 */ { CMD_CONV,  13, 13,512, 1024, 3,1,1, F_LR },
/* 23 */ { CMD_CONV,  13, 13,1024,1024, 3,1,1, F_LR },
/* 24 */ { CMD_CONV,  13, 13,1024,1024, 3,1,1, F_LR },
/* 25 */ { CMD_REORG, 26, 26, 512, 2048, 2,2,0, 0   }, /* REORG passthrough  */
/* 26 */ { CMD_CONV,  13, 13,3072,1024, 3,1,1, F_LR }, /* after concat       */
/* 27 */ { CMD_CONV,  13, 13,1024, 125, 1,1,0, 0    }, /* detection, linear  */
};
#define N_LAYERS  (sizeof(LAYERS)/sizeof(LAYERS[0]))

/* Layer 16 output (passthrough) is written to BUF1 and preserved there */
#define PASSTHROUGH_LAYER   16
#define REORG_LAYER         25
#define CONCAT_LAYER        26

/* Weight blob byte offsets in DDR – computed from layer shapes */
static uint32_t wt_offset[28];  /* byte offset from YOLOV2_WTS_BASE */

/* ============================================================
 * AXI-Lite register access
 * ============================================================ */
static volatile uint32_t *accel_reg = NULL;

static void reg_write(uint32_t off, uint32_t val) {
    accel_reg[off / 4] = val;
}
static uint32_t reg_read(uint32_t off) {
    return accel_reg[off / 4];
}

/* ============================================================
 * Accelerator launch helper
 * ============================================================ */
static void accel_run(const LayerCfg *l,
                      uint32_t in_addr, uint32_t out_addr, uint32_t wt_addr)
{
    reg_write(REG_CMD_OFF,      l->cmd);
    reg_write(REG_IN_ADDR_OFF,  in_addr);
    reg_write(REG_OUT_ADDR_OFF, out_addr);
    reg_write(REG_WT_ADDR_OFF,  wt_addr);
    reg_write(REG_IN_H_OFF,     l->in_h);
    reg_write(REG_IN_W_OFF,     l->in_w);
    reg_write(REG_IN_C_OFF,     l->in_c);
    reg_write(REG_OUT_C_OFF,    l->out_c);
    reg_write(REG_KSIZE_OFF,    l->ksize);
    reg_write(REG_STRIDE_OFF,   l->stride);
    reg_write(REG_PAD_OFF,      l->pad);
    reg_write(REG_FLAGS_OFF,    l->flags);

    /* Write ap_start (bit 0) */
    reg_write(REG_CTRL_OFF, 0x1);

    /* Poll ap_done (bit 1) */
    while (!(reg_read(REG_CTRL_OFF) & 0x2));
}

/* ============================================================
 * Weight offset table (call once at startup)
 * ============================================================ */
static void build_wt_offsets(void)
{
    uint32_t off = 0;
    for (int i = 0; i < (int)N_LAYERS; i++) {
        wt_offset[i] = off;
        if (LAYERS[i].cmd == CMD_CONV) {
            uint32_t n_wt = LAYERS[i].out_c * LAYERS[i].in_c
                           * LAYERS[i].ksize * LAYERS[i].ksize;
            uint32_t n_b  = LAYERS[i].out_c;
            off += (n_wt + n_b) * 2;   /* 2 bytes per INT16 value */
        }
    }
}

/* ============================================================
 * Public API
 * ============================================================ */

int yolov2_init(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return -1; }

    accel_reg = (volatile uint32_t *)mmap(NULL, 65536,
        PROT_READ | PROT_WRITE, MAP_SHARED, fd, YOLOV2_ACCEL_BASE);
    close(fd);
    if (accel_reg == MAP_FAILED) { perror("mmap accel"); return -1; }

    build_wt_offsets();
    printf("[yolov2] accelerator mapped at 0x%08X\n", YOLOV2_ACCEL_BASE);
    return 0;
}

void yolov2_cleanup(void)
{
    if (accel_reg && accel_reg != MAP_FAILED)
        munmap((void *)accel_reg, 65536);
    accel_reg = NULL;
}

int yolov2_load_weights(const char *weight_file)
{
    FILE *fp = fopen(weight_file, "rb");
    if (!fp) { perror(weight_file); return -1; }

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); fclose(fp); return -1; }

    /* Map enough DDR for all weights (~134 MB at INT16) */
    const size_t WTS_MAP_SIZE = 140 * 1024 * 1024;
    uint8_t *wt_ddr = (uint8_t *)mmap(NULL, WTS_MAP_SIZE,
        PROT_READ | PROT_WRITE, MAP_SHARED, fd, YOLOV2_WTS_BASE);
    close(fd);
    if (wt_ddr == MAP_FAILED) { perror("mmap weights"); fclose(fp); return -1; }

    size_t n = fread(wt_ddr, 1, WTS_MAP_SIZE, fp);
    fclose(fp);
    munmap(wt_ddr, WTS_MAP_SIZE);

    printf("[yolov2] loaded %.1f MB of weights from %s\n",
           n / 1048576.0f, weight_file);
    return 0;
}

int yolov2_infer(const uint8_t *rgb_416, int n_det_max, Detection *dets)
{
    /* ---- Quantise input image INT8 → INT16 fixed-point ------------------- */
    {
        int fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd < 0) { perror("open /dev/mem"); return -1; }
        const int IMG_BYTES = 416 * 416 * 3 * 2;
        int16_t *img_buf = (int16_t *)mmap(NULL, IMG_BYTES,
            PROT_READ | PROT_WRITE, MAP_SHARED, fd, YOLOV2_IMG_BASE);
        close(fd);
        if (img_buf == MAP_FAILED) { perror("mmap img"); return -1; }

        /* Normalise to [0,1] in ap_fixed<16,8> → scale = 1/255 * 256 = 1.004 */
        const float scale = 1.0f / 255.0f;
        int n = 416 * 416 * 3;
        for (int i = 0; i < n; i++) {
            float v = rgb_416[i] * scale;          /* [0, 1.0] */
            img_buf[i] = (int16_t)(v * 256.0f);    /* Q8.8 fixed-point */
        }
        munmap(img_buf, IMG_BYTES);
    }

    /* ---- Run layer sequence ---------------------------------------------- */
    uint32_t buf_in  = YOLOV2_IMG_BASE;
    uint32_t buf_out = YOLOV2_BUF0_BASE;
    uint32_t pt_addr = YOLOV2_BUF1_BASE;   /* passthrough held here */

    for (int i = 0; i < (int)N_LAYERS; i++) {
        const LayerCfg *l = &LAYERS[i];
        uint32_t wt_abs = YOLOV2_WTS_BASE + wt_offset[i];

        if (i == REORG_LAYER) {
            /* REORG: read layer-16 output from pt_addr, write to BUF1 end */
            uint32_t reorg_out = pt_addr + 26*26*512*2;  /* immediately after */
            accel_run(l, pt_addr, reorg_out, 0);
            /* Now concat: reorg_out (13×13×2048) is already placed such that
             * layer-24 output (in buf_in = BUF0) is at the start of buf_out;
             * we copy/alias: the concat buffer is BUF1 laid out as
             *   [0 .. 13*13*1024*2)   = layer-24 output  (copy from BUF0)
             *   [13*13*1024*2 ..)     = reorg output
             * For simplicity in bare-metal, just let layer-26 read from
             * a logically contiguous buffer that the PS has set up.
             * Here we update buf_in to point to that concat region.
             * (A DMA memcpy on the PS side achieves this efficiently.) */
            buf_in  = YOLOV2_BUF1_BASE;   /* concat buffer prepared by PS */
            buf_out = YOLOV2_BUF0_BASE;
            continue;  /* concat is done by addressing, no separate hw op */
        }

        if (i == CONCAT_LAYER) {
            /* buf_in already set to concat region by REORG handler above */
            uint32_t tmp = buf_out;
            accel_run(l, buf_in, tmp, wt_abs);
            buf_in  = tmp;
            buf_out = (tmp == YOLOV2_BUF0_BASE) ? YOLOV2_BUF1_BASE : YOLOV2_BUF0_BASE;
            continue;
        }

        /* Layers 0-24, 27 */
        accel_run(l, buf_in, buf_out, (l->cmd == CMD_CONV) ? wt_abs : 0);

        /* Save layer-16 output for passthrough before it gets overwritten */
        if (i == PASSTHROUGH_LAYER) {
            /* PS copies BUF0 → BUF1 (passthrough region) via memcpy */
            int fd = open("/dev/mem", O_RDWR | O_SYNC);
            if (fd >= 0) {
                size_t sz = 26*26*512*2;
                uint8_t *src = (uint8_t *)mmap(NULL, sz, PROT_READ,
                    MAP_SHARED, fd, buf_out);
                uint8_t *dst = (uint8_t *)mmap(NULL, sz, PROT_WRITE,
                    MAP_SHARED, fd, pt_addr);
                if (src != MAP_FAILED && dst != MAP_FAILED)
                    memcpy(dst, src, sz);
                if (src != MAP_FAILED) munmap(src, sz);
                if (dst != MAP_FAILED) munmap(dst, sz);
                close(fd);
            }
        }

        /* Ping-pong buffers */
        uint32_t tmp = buf_in;
        buf_in  = buf_out;
        buf_out = (i == 0) ? YOLOV2_BUF0_BASE : tmp;
        if (buf_out == buf_in)
            buf_out = (buf_in == YOLOV2_BUF0_BASE) ? YOLOV2_BUF1_BASE : YOLOV2_BUF0_BASE;
    }

    /* Final output is in buf_in */
    /* Copy result to OUT_BASE (optional, for clean API) */
    printf("[yolov2] inference done, decoding detections...\n");
    return yolov2_decode(YOLOV2_THRESH, YOLOV2_NMS_THRESH, n_det_max, dets);
}

/* ============================================================
 * Detection decode: sigmoid / softmax / NMS
 * ============================================================ */
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static float iou(const Detection *a, const Detection *b) {
    float ax0 = a->x - a->w*0.5f, ax1 = a->x + a->w*0.5f;
    float ay0 = a->y - a->h*0.5f, ay1 = a->y + a->h*0.5f;
    float bx0 = b->x - b->w*0.5f, bx1 = b->x + b->w*0.5f;
    float by0 = b->y - b->h*0.5f, by1 = b->y + b->h*0.5f;
    float ix = fmaxf(0.f, fminf(ax1,bx1) - fmaxf(ax0,bx0));
    float iy = fmaxf(0.f, fminf(ay1,by1) - fmaxf(ay0,by0));
    float inter = ix * iy;
    float ua = a->w*a->h + b->w*b->h - inter;
    return (ua > 0) ? inter/ua : 0;
}

int yolov2_decode(float thresh, float nms_thresh,
                  int n_det_max, Detection *dets)
{
    /* Map final output buffer */
    int fd = open("/dev/mem", O_RDONLY | O_SYNC);
    if (fd < 0) return -1;

    const int OUT_ELEMS = YOLOV2_GRID_H * YOLOV2_GRID_W
                        * YOLOV2_ANCHORS * (5 + YOLOV2_CLASSES);
    int16_t *raw = (int16_t *)mmap(NULL, OUT_ELEMS * 2,
        PROT_READ, MAP_SHARED, fd, YOLOV2_OUT_BASE);
    close(fd);
    if (raw == MAP_FAILED) return -1;

    int n_dets = 0;
    const float SCALE = 1.0f / 256.0f;   /* Q8.8 → float */

    for (int a = 0; a < YOLOV2_ANCHORS && n_dets < n_det_max; a++) {
        for (int gy = 0; gy < YOLOV2_GRID_H && n_dets < n_det_max; gy++) {
            for (int gx = 0; gx < YOLOV2_GRID_W && n_dets < n_det_max; gx++) {
                int base = (a * (5 + YOLOV2_CLASSES))
                           * YOLOV2_GRID_H * YOLOV2_GRID_W
                           + gy * YOLOV2_GRID_W + gx;
                int stride = YOLOV2_GRID_H * YOLOV2_GRID_W;

                float tx  = raw[base + 0*stride] * SCALE;
                float ty  = raw[base + 1*stride] * SCALE;
                float tw  = raw[base + 2*stride] * SCALE;
                float th  = raw[base + 3*stride] * SCALE;
                float obj = sigmoid(raw[base + 4*stride] * SCALE);

                if (obj < thresh) continue;

                /* Softmax over classes */
                float cls[YOLOV2_CLASSES];
                float max_cls = -1e9f;
                for (int c=0; c<YOLOV2_CLASSES; c++) {
                    cls[c] = raw[base + (5+c)*stride] * SCALE;
                    if (cls[c] > max_cls) max_cls = cls[c];
                }
                float sum = 0;
                for (int c=0; c<YOLOV2_CLASSES; c++) {
                    cls[c] = expf(cls[c] - max_cls);
                    sum += cls[c];
                }
                int best = 0;
                float best_score = 0;
                for (int c=0; c<YOLOV2_CLASSES; c++) {
                    cls[c] /= sum;
                    float sc = obj * cls[c];
                    if (sc > best_score) { best_score = sc; best = c; }
                    dets[n_dets].cls_scores[c] = cls[c];
                }
                if (best_score < thresh) continue;

                dets[n_dets].x   = (gx + sigmoid(tx)) / YOLOV2_GRID_W * 416.0f;
                dets[n_dets].y   = (gy + sigmoid(ty)) / YOLOV2_GRID_H * 416.0f;
                dets[n_dets].w   = YOLOV2_ANCHOR_W[a] * expf(tw) / YOLOV2_GRID_W * 416.0f;
                dets[n_dets].h   = YOLOV2_ANCHOR_H[a] * expf(th) / YOLOV2_GRID_H * 416.0f;
                dets[n_dets].obj = obj;
                dets[n_dets].cls = best;
                n_dets++;
            }
        }
    }
    munmap(raw, OUT_ELEMS * 2);

    /* Greedy NMS per class */
    for (int c = 0; c < YOLOV2_CLASSES; c++) {
        for (int i = 0; i < n_dets-1; i++) {
            if (dets[i].cls != c) continue;
            for (int j = i+1; j < n_dets; j++) {
                if (dets[j].cls != c) continue;
                if (iou(&dets[i], &dets[j]) > nms_thresh) {
                    /* suppress the lower-scoring box */
                    if (dets[i].obj < dets[j].obj)
                        dets[i] = dets[--n_dets];
                    else
                        dets[j] = dets[--n_dets];
                }
            }
        }
    }

    printf("[yolov2] %d detections after NMS\n", n_dets);
    return n_dets;
}
