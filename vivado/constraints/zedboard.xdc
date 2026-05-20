## ============================================================
## zedboard.xdc  –  ZedBoard (XC7Z020CLG484-1) constraints
## ============================================================

# ---- Primary clock (PS fabric clock FCLK0 at 100 MHz) --------------------
# The PS7 drives FCLK_CLK0 internally; create a virtual clock for timing
# analysis of the PL logic it drives.
create_clock -period 10.000 -name fclk0 -waveform {0.000 5.000} \
    [get_ports {ps7_0_FCLK_CLK0}]

# ---- PS fixed IO -----------------------------------------------------------
set_property PACKAGE_PIN   N7  [get_ports {DDR_CAS_n}]
set_property IOSTANDARD    SSTL15 [get_ports {DDR_CAS_n}]

# (Full pin assignments are provided by the Zynq PS7 IP and board preset.
#  Add custom PL pin assignments below if needed.)

# ---- False path / multicycle exceptions ------------------------------------
# AXI-Lite config path: data arrives many cycles before fabric processes it.
set_multicycle_path 2 -setup -from [get_cells -hier -filter {NAME =~ *s_axi_ctrl*}]

# ---- Bitstream settings ----------------------------------------------------
set_property BITSTREAM.GENERAL.COMPRESS   TRUE  [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH  4    [current_design]
set_property CONFIG_VOLTAGE               3.3   [current_design]
set_property CFGBVS                       VCCO  [current_design]
