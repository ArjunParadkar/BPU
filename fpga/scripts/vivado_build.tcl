# vivado_build.tcl -- synthesis, implementation, bitstream, timing + utilisation reports.  Non-interactive:
#   vivado -mode batch -source scripts/vivado_build.tcl -tclargs <out_dir>
# Produces <out_dir>/bpu_kv260.runs/impl_1/bd_wrapper.bit, timing_summary.rpt, utilization.rpt, power.rpt and
# <out_dir>/results.json (Fmax from the achieved WNS, DSP/BRAM/LUT counts) for docs/RESULTS.md.
set out [lindex $argv 0]
open_project $out/bpu_kv260.xpr
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1
report_timing_summary -file $out/timing_summary.rpt
report_utilization -hierarchical -file $out/utilization.rpt
report_power -file $out/power.rpt
set wns [get_property STATS.WNS [get_runs impl_1]]
set period [get_property PERIOD [get_clocks -of_objects [get_pins -hier *tile*/aclk] -quiet]]
if {$period eq ""} { set period 5.0 }
set fmax [expr {1000.0 / ($period - $wns)}]
set f [open $out/results.json w]
puts $f "{\"wns_ns\": $wns, \"clock_period_ns\": $period, \"fmax_mhz\": $fmax, \"dsp\": [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ ARITHMETIC.DSP.*}]], \"bram36\": [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ BLOCKRAM.BRAM.RAMB36*}]], \"bram18\": [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ BLOCKRAM.BRAM.RAMB18*}]], \"luts\": [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ CLB.LUT.*}]], \"timing_closed\": [expr {$wns >= 0 ? \"true\" : \"false\"}]}"
close $f
write_hw_platform -fixed -include_bit -force $out/bpu_kv260.xsa
puts "BUILD_DONE"
