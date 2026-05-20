## ============================================================
## create_project.tcl
## Creates the Vivado block-design project for YOLOv2 on ZedBoard.
##
## Prerequisites:
##   1. Run run_hls.tcl first to export the HLS IP.
##   2. Vivado 2019.2+ installed and 'vivado' on PATH.
##
## Usage:
##   cd <repo>/hls/scripts
##   vivado -mode tcl -source create_project.tcl
## ============================================================

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize "$script_dir/../.."]

# ---- Create project ---------------------------------------------------------
set proj_dir "$repo_root/vivado/yolov2_zedboard"
create_project yolov2_zedboard $proj_dir -part xc7z020clg484-1 -force

set_property board_part em.avnet.com:zed:part0:1.4 [current_project]

# ---- Register HLS IP repository --------------------------------------------
set ip_repo "$repo_root/hls/yolov2_hls_prj/solution1/impl/ip"
if {![file isdirectory $ip_repo]} {
    puts "ERROR: HLS IP not found at $ip_repo"
    puts "       Run 'vivado_hls -f run_hls.tcl' first."
    exit 1
}
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

# ---- Block Design -----------------------------------------------------------
create_bd_design "yolov2_design"
update_compile_order -fileset sources_1

# Zynq PS7
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} \
    [get_bd_cells ps7_0]

# Enable AXI HP0 port for DMA (high-performance slave → DDR)
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0  {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
] [get_bd_cells ps7_0]

# YOLOv2 HLS accelerator IP
create_bd_cell -type ip \
    -vlnv user.org:user:yolov2_accel:1.0 yolov2_accel_0

# AXI Interconnect: PS M_AXI_GP0 → yolov2 s_axi_ctrl  (AXI-Lite)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ic_ctrl
set_property CONFIG.NUM_SI {1} [get_bd_cells axi_ic_ctrl]
set_property CONFIG.NUM_MI {1} [get_bd_cells axi_ic_ctrl]

# AXI Interconnect: yolov2 m_axi_gmem → PS S_AXI_HP0  (AXI full)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ic_data
set_property CONFIG.NUM_SI {1} [get_bd_cells axi_ic_data]
set_property CONFIG.NUM_MI {1} [get_bd_cells axi_ic_data]

# ---- Clocking ---------------------------------------------------------------
# FCLK_CLK0 drives both PS GP0 and accelerator
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] \
    [get_bd_pins yolov2_accel_0/ap_clk] \
    [get_bd_pins axi_ic_ctrl/ACLK]     \
    [get_bd_pins axi_ic_ctrl/S00_ACLK] \
    [get_bd_pins axi_ic_ctrl/M00_ACLK] \
    [get_bd_pins axi_ic_data/ACLK]     \
    [get_bd_pins axi_ic_data/S00_ACLK] \
    [get_bd_pins axi_ic_data/M00_ACLK]

# ---- Resets -----------------------------------------------------------------
connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] \
    [get_bd_pins yolov2_accel_0/ap_rst_n] \
    [get_bd_pins axi_ic_ctrl/ARESETN]     \
    [get_bd_pins axi_ic_ctrl/S00_ARESETN] \
    [get_bd_pins axi_ic_ctrl/M00_ARESETN] \
    [get_bd_pins axi_ic_data/ARESETN]     \
    [get_bd_pins axi_ic_data/S00_ARESETN] \
    [get_bd_pins axi_ic_data/M00_ARESETN]

# ---- AXI connections --------------------------------------------------------
# PS GP0 → ctrl interconnect → yolov2 ctrl slave
connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0] \
    [get_bd_intf_pins axi_ic_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ic_ctrl/M00_AXI] \
    [get_bd_intf_pins yolov2_accel_0/s_axi_ctrl]

# yolov2 gmem master → data interconnect → PS HP0
connect_bd_intf_net [get_bd_intf_pins yolov2_accel_0/m_axi_gmem] \
    [get_bd_intf_pins axi_ic_data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ic_data/M00_AXI] \
    [get_bd_intf_pins ps7_0/S_AXI_HP0]

# Interrupt: ap_done → PS IRQ (optional)
connect_bd_net [get_bd_pins yolov2_accel_0/interrupt] \
    [get_bd_pins ps7_0/IRQ_F2P]

# ---- Address map (PS sees accelerator ctrl at 0x4000_0000) -----------------
create_bd_addr_seg \
    -range 64K -offset 0x40000000 \
    [get_bd_addr_spaces ps7_0/Data] \
    [get_bd_addr_segs yolov2_accel_0/s_axi_ctrl/Reg] \
    SEG_yolov2_ctrl

# Accelerator sees full 512 MB PS DDR
create_bd_addr_seg \
    -range 512M -offset 0x00000000 \
    [get_bd_addr_spaces yolov2_accel_0/Data_m_axi_gmem] \
    [get_bd_addr_segs ps7_0/S_AXI_HP0/HP0_DDR_LOWOCM] \
    SEG_ps_ddr

# ---- Validate & generate ---------------------------------------------------
validate_bd_design
make_wrapper -files [get_files yolov2_design.bd] -top
add_files -norecurse \
    "$proj_dir/yolov2_zedboard.srcs/sources_1/bd/yolov2_design/hdl/yolov2_design_wrapper.v"
set_property top yolov2_design_wrapper [current_fileset]

# ---- Constraints -----------------------------------------------------------
add_files -fileset constrs_1 "$repo_root/vivado/constraints/zedboard.xdc"

# ---- Synthesis & Implementation (optional) --------------------------------
# Uncomment to run full implementation flow:
# launch_runs synth_1 -jobs 4
# wait_on_run synth_1
# launch_runs impl_1 -to_step write_bitstream -jobs 4
# wait_on_run impl_1
# write_hw_platform -fixed -force -file "$proj_dir/yolov2_zedboard.xsa"

puts ""
puts "Vivado project created at: $proj_dir"
puts "Open in GUI with:  vivado $proj_dir/yolov2_zedboard.xpr"
puts "Then run: Flow > Generate Bitstream"
