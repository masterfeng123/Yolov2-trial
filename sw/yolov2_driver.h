#pragma once

#include <stdint.h>
#include <stdbool.h>

/*
 * YOLOv2 PS-side driver
 *
 * Sequences the 28-layer YOLOv2 inference pass by writing configuration
 * registers into the HLS accelerator's AXI-Lite slave and polling ap_done.
 *
 * Memory layout in DDR (byte addresses, adjustable via YOLOV2_*_BASE):
 *
 *   YOLOV2_WTS_BASE   : pre-loaded YOLOv2 weights (folded BN, INT16)  ~134 MB
 *   YOLOV2_BUF0_BASE  : feature-map ping buffer                        ~8 MB
 *   YOLOV2_BUF1_BASE  : feature-map pong buffer / reorg staging        ~8 MB
 *   YOLOV2_IMG_BASE   : input image (416×416×3 INT16)                  ~0.5 MB
 *   YOLOV2_OUT_BASE   : detection output (13×13×125 INT16)             ~0.04 MB
 */

/* AXI-Lite base address of the HLS accelerator (set by block-design) */
#define YOLOV2_ACCEL_BASE   0x40000000UL

/* DDR region layout */
#define YOLOV2_WTS_BASE     0x10000000UL   /* 256 MB mark – weights     */
#define YOLOV2_BUF0_BASE    0x18A00000UL   /* after weights             */
#define YOLOV2_BUF1_BASE    0x19200000UL
#define YOLOV2_IMG_BASE     0x19A00000UL
#define YOLOV2_OUT_BASE     0x19B00000UL

/* AXI-Lite register offsets (Vivado HLS auto-generates these) */
#define REG_CTRL_OFF        0x00u   /* bit0=ap_start, bit1=ap_done, bit2=ap_idle */
#define REG_GIE_OFF         0x04u   /* global interrupt enable                   */
#define REG_IER_OFF         0x08u   /* interrupt enable                          */
#define REG_ISR_OFF         0x0Cu   /* interrupt status                          */
#define REG_CMD_OFF         0x10u
#define REG_IN_ADDR_OFF     0x18u
#define REG_OUT_ADDR_OFF    0x20u
#define REG_WT_ADDR_OFF     0x28u
#define REG_IN_H_OFF        0x30u
#define REG_IN_W_OFF        0x38u
#define REG_IN_C_OFF        0x40u
#define REG_OUT_C_OFF       0x48u
#define REG_KSIZE_OFF       0x50u
#define REG_STRIDE_OFF      0x58u
#define REG_PAD_OFF         0x60u
#define REG_FLAGS_OFF       0x68u

/* Command codes */
#define CMD_IDLE            0x00u
#define CMD_CONV            0x01u
#define CMD_POOL            0x02u
#define CMD_REORG           0x03u

/* Flags */
#define FLAG_LRELU          0x01u

/* YOLOv2 VOC detection constants */
#define YOLOV2_CLASSES      20
#define YOLOV2_ANCHORS      5
#define YOLOV2_GRID_H       13
#define YOLOV2_GRID_W       13
#define YOLOV2_THRESH       0.3f    /* object-confidence threshold */
#define YOLOV2_NMS_THRESH   0.45f

/* Anchor prior sizes (biased toward COCO scale, normalised to 32-pixel grid) */
static const float YOLOV2_ANCHOR_W[5] = {1.3221f, 3.1923f, 5.5556f, 9.8028f, 3.6518f};
static const float YOLOV2_ANCHOR_H[5] = {1.7314f, 4.0094f, 4.3655f, 9.0094f, 3.6833f};

/* Detection box result */
typedef struct {
    float x, y, w, h;   /* centre-x/y, width, height – in pixel units */
    float obj;           /* objectness score */
    float cls_scores[20];
    int   cls;           /* argmax class */
} Detection;

/* ---- Public API --------------------------------------------------------- */

/*
 * Initialise the driver: maps ACCEL_BASE to a virtual address using mmap.
 * Returns 0 on success.
 */
int  yolov2_init(void);
void yolov2_cleanup(void);

/*
 * Load BN-folded INT16 weights from a binary file to DDR.
 * weight_file: path to yolov2-voc-folded-int16.bin
 */
int  yolov2_load_weights(const char *weight_file);

/*
 * Run full 28-layer inference on a 416×416 RGB image (uint8 pixels).
 * Writes INT16 quantised data to YOLOV2_IMG_BASE before starting.
 */
int  yolov2_infer(const uint8_t *rgb_416, int n_det_max, Detection *dets);

/*
 * Decode raw accelerator output (at YOLOV2_OUT_BASE) into Detection structs.
 * Returns number of detections after NMS.
 */
int  yolov2_decode(float thresh, float nms_thresh,
                   int n_det_max, Detection *dets);
