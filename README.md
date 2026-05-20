# YOLOv2 HLS Accelerator for ZedBoard

Implements YOLOv2 (Darknet-19, VOC 20-class) as a Vivado HLS IP core
targeting the **ZedBoard** (Xilinx Zynq XC7Z020CLG484-1).

---

## Architecture Overview

```
PS ARM Cortex-A9
  │  AXI-Lite (ctrl)        AXI4 master (gmem, 64-bit)
  ▼                                   │
 [yolov2_accel HLS IP] ◄──────────── DDR3 (512 MB)
   ├── conv_engine   (3×3 / 1×1 tiled convolution + bias + optional LeakyReLU)
   ├── pool_engine   (2×2 max-pool, stride 2)
   └── reorg_engine  (space-to-depth, stride 2)
```

The PS ARM controls inference layer-by-layer:

1. Write config registers to the AXI-Lite slave.
2. Set `ap_start` → poll `ap_done`.
3. Swap ping-pong feature-map buffers.
4. Post-process detections (sigmoid / softmax / NMS) in software.

---

## YOLOv2 Layer Sequence (28 layers, VOC)

| # | Type    | Input         | Output        | Kernel | Notes             |
|---|---------|---------------|---------------|--------|-------------------|
| 0 | CONV    | 416×416×3     | 416×416×32    | 3×3    | BN + LeakyReLU    |
| 1 | POOL    | 416×416×32    | 208×208×32    | 2×2    | stride 2          |
| 2 | CONV    | 208×208×32    | 208×208×64    | 3×3    |                   |
| 3 | POOL    | 208×208×64    | 104×104×64    | 2×2    |                   |
|16 | CONV    | 26×26×256     | 26×26×512     | 3×3    | → passthrough     |
|17 | POOL    | 26×26×512     | 13×13×512     | 2×2    |                   |
|24 | CONV    | 13×13×1024    | 13×13×1024    | 3×3    |                   |
|25 | REORG   | 26×26×512     | 13×13×2048    | –      | passthrough       |
|26 | CONV    | 13×13×3072    | 13×13×1024    | 3×3    | after concat      |
|27 | CONV    | 13×13×1024    | 13×13×125     | 1×1    | linear (no ReLU)  |

---

## Fixed-Point Quantisation

| Symbol   | Type              | Range   | Notes                 |
|----------|-------------------|---------|-----------------------|
| `data_t` | `ap_fixed<16,8>`  | ±128    | activations           |
| `wt_t`   | `ap_fixed<16,8>`  | ±128    | weights (BN-folded)   |
| `acc_t`  | `ap_fixed<32,16>` | ±32768  | accumulator           |

Batch normalisation is **folded offline**: `w' = w·γ/√(σ²+ε)`, `b' = β−γ·μ/√(σ²+ε)`

---

## Resource Estimate (XC7Z020)

| Resource | Budget  | Estimate          |
|----------|---------|-------------------|
| DSP48    | 220     | ~8 parallel MACs  |
| BRAM 36K | 140     | ~10               |
| LUT      | 53 200  | ~15 000           |

---

## Build Instructions

### Step 1 – HLS synthesis

```bash
cd hls/scripts
vitis_hls -f run_hls.tcl   # or vivado_hls for 2019.x
```

### Step 2 – Vivado block-design project

```bash
vivado -mode tcl -source hls/scripts/create_project.tcl
# Then open vivado/yolov2_zedboard/yolov2_zedboard.xpr → Generate Bitstream
```

### Step 3 – Convert weights (offline, on host PC)

```bash
python3 tools/convert_weights.py yolov2-voc.weights yolov2-voc-folded-int16.bin
```

### Step 4 – Run on ZedBoard (Linux)

```c
#include "sw/yolov2_driver.h"
yolov2_init();
yolov2_load_weights("yolov2-voc-folded-int16.bin");
Detection dets[100];
int n = yolov2_infer(rgb_image_416x416, 100, dets);
```

---

## Project Structure

```
hls/src/          – HLS C++ sources (conv / pool / reorg engines + top)
hls/tb/           – C simulation testbench
hls/scripts/      – run_hls.tcl, create_project.tcl
vivado/constraints/zedboard.xdc
sw/               – PS-side driver (layer sequencer + NMS decoder)
```
