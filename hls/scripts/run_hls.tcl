## ============================================================
## run_hls.tcl
## Vivado HLS (2019.2+) / Vitis HLS (2020.2+) build script
## Target: XC7Z020CLG484-1  (ZedBoard)
## Clock:  100 MHz  (10 ns period)
##
## Usage:
##   vivado_hls -f run_hls.tcl          (Vivado HLS)
##   vitis_hls  -f run_hls.tcl          (Vitis HLS)
## ============================================================

# ---- Project setup ---------------------------------------------------------
open_project yolov2_hls_prj
set_top yolov2_accel

# Source files
add_files [list \
    ../src/yolov2_accel.cpp \
    ../src/conv_engine.cpp  \
    ../src/pool_engine.cpp  \
    ../src/reorg_engine.cpp ]

# Testbench (not synthesised)
add_files -tb ../tb/tb_yolov2.cpp

# ---- Solution ---------------------------------------------------------------
open_solution "solution1" -reset
set_part {xc7z020clg484-1}
create_clock -period 10 -name default    ;# 100 MHz

# ---- Interface directives ---------------------------------------------------
# These mirror the pragmas inside yolov2_accel.cpp but can also be set here.
# set_directive_interface -mode m_axi     yolov2_accel gmem -bundle gmem \
#     -offset slave -depth 16777216 \
#     -max_read_burst_length 256 -max_write_burst_length 256
# set_directive_interface -mode s_axilite yolov2_accel -bundle ctrl

# ---- Optimisation hints (optional, tune after initial synthesis) -----------
# Increase unroll factor for the inner CSTRIP_O strip if DSP budget allows:
# set_directive_unroll -factor 8 yolov2_accel/INNER_OC_UNROLL

# ---- Synthesis --------------------------------------------------------------
csynth_design

# ---- RTL simulation (cosim) ------------------------------------------------
# Uncomment to run co-simulation after synthesis.
# cosim_design -O -rtl verilog -tool xsim

# ---- Export IP for Vivado block-design use ----------------------------------
export_design \
    -format    ip_catalog \
    -vendor    "user.org" \
    -library   "user" \
    -ipname    "yolov2_accel" \
    -version   "1.0" \
    -description "YOLOv2 convolution / pool / reorg accelerator for ZedBoard"

puts "HLS synthesis complete.  IP exported to:"
puts "  [pwd]/yolov2_hls_prj/solution1/impl/ip"
