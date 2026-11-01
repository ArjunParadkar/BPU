// shift_add: differential readout and partial-sum combination across passes (contract section 3 steps 6-7).
//   Y[c] += (I+[c] - I-[c]) << (8 * pass)
module shift_add
  import bpu_pkg::*;
(
  input  logic                      clk,
  input  logic                      clear,       // start of a read
  input  logic                      en,          // end of a pass: fold the pass currents in
  input  logic                      pass_idx,
  input  logic [COLS*CUR_BITS-1:0]  i_p,
  input  logic [COLS*CUR_BITS-1:0]  i_m,
  output logic [COLS*ACC_BITS-1:0]  y_acc        // 32 x signed 40
);
  genvar c;
  generate
    for (c = 0; c < COLS; c++) begin : g_col
      logic signed [ACC_BITS-1:0] y;
      logic signed [CUR_BITS-1:0] diff;
      logic signed [ACC_BITS-1:0] addend;
      always_comb begin
        diff   = $signed(i_p[c*CUR_BITS +: CUR_BITS]) - $signed(i_m[c*CUR_BITS +: CUR_BITS]);
        addend = pass_idx ? (ACC_BITS'(diff) <<< 8) : ACC_BITS'(diff);
      end
      always_ff @(posedge clk) begin
        if (clear)   y <= '0;
        else if (en) y <= y + addend;
      end
      assign y_acc[c*ACC_BITS +: ACC_BITS] = y;
    end
  endgenerate
endmodule
