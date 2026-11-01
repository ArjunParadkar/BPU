# kv260.xdc -- the PL clock is pl_clk0 from the PS (200 MHz requested in vivado_bd.tcl); the PS IP creates the
# clock object, so no create_clock here.  No external I/O: the tile talks to the PS only.
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
