// col_integrate: per-column exact accumulation of the row terms over one pass (contract section 3 step 5).
// clear  : zero the integrators (start of a pass)
// en     : add this cycle's terms
module col_integrate
  import bpu_pkg::*;
(
  input  logic                        clk,
  input  logic                        clear,
  input  logic                        en,
  input  logic [COLS*TERM_BITS-1:0]   term_p,
  input  logic [COLS*TERM_BITS-1:0]   term_m,
  output logic [COLS*CUR_BITS-1:0]    i_p,       // 32 x signed 32
  output logic [COLS*CUR_BITS-1:0]    i_m
);
  genvar c;
  generate
    for (c = 0; c < COLS; c++) begin : g_col
      logic signed [CUR_BITS-1:0] ap, am;
      always_ff @(posedge clk) begin
        if (clear) begin ap <= '0; am <= '0; end
        else if (en) begin
          ap <= ap + CUR_BITS'($signed(term_p[c*TERM_BITS +: TERM_BITS]));
          am <= am + CUR_BITS'($signed(term_m[c*TERM_BITS +: TERM_BITS]));
        end
      end
      assign i_p[c*CUR_BITS +: CUR_BITS] = ap;
      assign i_m[c*CUR_BITS +: CUR_BITS] = am;
    end
  endgenerate
endmodule
