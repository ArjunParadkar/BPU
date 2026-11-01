# vivado_bd.tcl -- KV260 block design: Zynq UltraScale+ PS + AXI DMA + bpu_tile_axi.  Non-interactive:
#   vivado -mode batch -source scripts/vivado_bd.tcl -tclargs <out_dir>
# Addresses used by ps/src/hal_mmap.c: tile 0xA000_0000 (64 KiB), DMA 0xA001_0000 (64 KiB).
# Tool version pinned in scripts/VERSIONS.md.  NOT YET RUN HERE (no Vivado on the bench, 2026-11-01).
set out [lindex $argv 0]
set hdl [file normalize [file dirname [info script]]/../hdl]
create_project bpu_kv260 $out -part xck26-sfvc784-2LV-c -force
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]
add_files [glob $hdl/*.sv]
add_files -norecurse $hdl/zlut_q13.mem
set_property file_type SystemVerilog [get_files *.sv]
set_property top bpu_tile_axi [current_fileset]
update_compile_order -fileset sources_1
create_bd_design "bd"
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells ps]
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} CONFIG.PSU__SAXIGP2__DATA_WIDTH {64} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200}] [get_bd_cells ps]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_include_stscntrl_strm {0} CONFIG.c_m_axis_mm2s_tdata_width {64} \
  CONFIG.c_m_axi_mm2s_data_width {64} CONFIG.c_s_axis_s2mm_tdata_width {64} CONFIG.c_m_axi_s2mm_data_width {64} \
  CONFIG.c_mm2s_burst_size {16} CONFIG.c_s2mm_burst_size {16} CONFIG.c_sg_length_width {23}] [get_bd_cells dma]
create_bd_cell -type module -reference bpu_tile_axi tile
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_pins tile/s_axis]
connect_bd_intf_net [get_bd_intf_pins tile/m_axis] [get_bd_intf_pins dma/S_AXIS_S2MM]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/ps/pl_clk0} Clk_slave {Auto} Clk_xbar {Auto} Master {/ps/M_AXI_HPM0_FPD} Slave {/tile/s_axil} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}} [get_bd_intf_pins tile/s_axil]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/ps/pl_clk0} Clk_slave {Auto} Clk_xbar {Auto} Master {/ps/M_AXI_HPM0_FPD} Slave {/dma/S_AXI_LITE} ddr_seg {Auto} intc_ip {/ps_axi_periph} master_apm {0}} [get_bd_intf_pins dma/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/ps/pl_clk0} Clk_slave {Auto} Clk_xbar {Auto} Master {/dma/M_AXI_MM2S} Slave {/ps/S_AXI_HP0_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}} [get_bd_intf_pins ps/S_AXI_HP0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {/ps/pl_clk0} Clk_slave {Auto} Clk_xbar {Auto} Master {/dma/M_AXI_S2MM} Slave {/ps/S_AXI_HP0_FPD} ddr_seg {Auto} intc_ip {/axi_smc} master_apm {0}} [get_bd_intf_pins dma/M_AXI_S2MM]
assign_bd_address
set_property offset 0xA0000000 [get_bd_addr_segs {ps/Data/SEG_tile_reg0}]
set_property range 64K [get_bd_addr_segs {ps/Data/SEG_tile_reg0}]
set_property offset 0xA0010000 [get_bd_addr_segs {ps/Data/SEG_dma_Reg}]
validate_bd_design
save_bd_design
make_wrapper -files [get_files bd.bd] -top -import
set_property top bd_wrapper [current_fileset]
add_files -fileset constrs_1 [file normalize [file dirname [info script]]/kv260.xdc]
puts "BD_DONE"
