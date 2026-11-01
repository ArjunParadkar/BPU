// wl_dac: activation code -> row drive for the current pass (contract section 2/3 step 1).
//   passes = 1 : d = x[7:0] as int8
//   passes = 2 : pass 0 -> low byte unsigned [0,255]; pass 1 -> high byte int8   (x = 256*d1 + d0)
module wl_dac
  import bpu_pkg::*;
(
  input  logic [15:0]                  x_code,
  input  logic                         two_pass,
  input  logic                         pass_idx,
  output logic signed [DRIVE_BITS-1:0] d
);
  always_comb begin
    if (!two_pass)            d = {x_code[7], x_code[7:0]};        // sign-extend int8
    else if (pass_idx == 1'b0) d = {1'b0, x_code[7:0]};            // unsigned low byte
    else                      d = {x_code[15], x_code[15:8]};      // signed high byte
  end
endmodule
