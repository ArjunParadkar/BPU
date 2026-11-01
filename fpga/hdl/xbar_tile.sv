// xbar_tile: one row of the 32x32 differential array per clock (contract section 3 steps 2-5, products only).
// For every column c: term_p[c] = d * A[c] * Ghat(w_p[c], eta_p[c]),  term_m[c] likewise.
// Two pipeline stages (ghat register, product register): terms appear 2 clocks after the inputs.
// 128 multipliers -> DSP48E2.
module xbar_tile
  import bpu_pkg::*;
(
  input  logic                         clk,
  input  logic signed [DRIVE_BITS-1:0] d,
  input  logic [COLS*W_BITS-1:0]       w_p,       // 32 x 8
  input  logic [COLS*W_BITS-1:0]       w_m,
  input  logic [COLS-1:0]              mask,
  input  logic [COLS*16-1:0]           eta_p,     // 32 x int16
  input  logic [COLS*16-1:0]           eta_m,
  output logic [COLS*TERM_BITS-1:0]    term_p,    // 32 x signed 27
  output logic [COLS*TERM_BITS-1:0]    term_m
);
  genvar c;
  generate
    for (c = 0; c < COLS; c++) begin : g_col
      logic [GHAT_BITS-1:0] gp, gm, gp_r, gm_r;
      logic signed [DRIVE_BITS-1:0] d_r;
      gcond up (.w_q(w_p[c*W_BITS +: W_BITS]), .eta_q(eta_p[c*16 +: 16]), .ghat(gp));
      gcond um (.w_q(w_m[c*W_BITS +: W_BITS]), .eta_q(eta_m[c*16 +: 16]), .ghat(gm));
      always_ff @(posedge clk) begin
        gp_r <= mask[c] ? gp : '0;
        gm_r <= mask[c] ? gm : '0;
        d_r  <= d;
        term_p[c*TERM_BITS +: TERM_BITS] <= d_r * $signed({1'b0, gp_r});
        term_m[c*TERM_BITS +: TERM_BITS] <= d_r * $signed({1'b0, gm_r});
      end
    end
  endgenerate
endmodule
