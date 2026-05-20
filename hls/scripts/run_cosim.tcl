## ============================================================
## run_cosim.tcl  –  Run Vivado HLS C-simulation + co-simulation
##
## Prerequisites:
##   1. tools/verify/run_verification.py has been executed to
##      generate hls/tb/cosim_data/
##   2. Vivado HLS / Vitis HLS installed
##
## Usage:
##   cd hls/scripts
##   vivado_hls -f run_cosim.tcl
##   vitis_hls  -f run_cosim.tcl
## ============================================================

# ---- Project (re-opens if already exists) -----------------------------------
open_project yolov2_hls_prj

set_top yolov2_accel

add_files [list \
    ../src/yolov2_accel.cpp \
    ../src/conv_engine.cpp  \
    ../src/pool_engine.cpp  \
    ../src/reorg_engine.cpp ]

# Use the file-driven co-sim testbench
add_files -tb [list \
    ../tb/tb_cosim.cpp      \
    ../tb/cosim_utils.h     ]

# Pass the cosim_data path to the testbench executable
set_property -name cosim_argv -value "../../hls/tb/cosim_data" -object [current_solution]

# ---- Solution ---------------------------------------------------------------
open_solution "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10 -name default

# ---- Step 1: C simulation  (pure software, fast) ----------------------------
puts "\n[cosim] Running C simulation ..."
csim_design -O -argv "../../hls/tb/cosim_data"

# ---- Step 2: HLS synthesis (required before co-sim) -------------------------
puts "\n[cosim] Running C synthesis ..."
csynth_design

# ---- Step 3: RTL co-simulation (actual hardware simulation) -----------------
puts "\n[cosim] Running RTL co-simulation (Vivado xsim) ..."
cosim_design \
    -O              \
    -rtl    verilog \
    -tool   xsim    \
    -argv   "../../hls/tb/cosim_data"

# ---- Results ----------------------------------------------------------------
puts ""
puts "Co-simulation complete."
puts "HLS outputs written to: hls/tb/cosim_data/<layer>/hls_output.bin"
puts "Run comparison:  python3 tools/verify/layer_compare.py --hls_cosim"
