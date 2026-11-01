// gcond: state code + noise sample -> noisy conductance code (contract section 3 steps 2-3).
//   G_q    = 655 + 254 * w_q                          (16 bits, max 65425)
//   G_hat  = (G_q * (16384 + eta_q) + 8192) >> 14     (17 bits, exact, no saturation)
module gcond
  import bpu_pkg::*;
(
  input  logic [W_BITS-1:0]     w_q,
  input  logic signed [15:0]    eta_q,
  output logic [GHAT_BITS-1:0]  ghat
);
  logic [15:0] g;
  logic [15:0] gain;      // 1 .. 32767
  logic [31:0] prod;
  logic [32:0] rounded;
  always_comb begin
    g       = 16'd655 + {w_q, 8'b0} - {7'b0, w_q, 1'b0};
    gain    = 16'(17'sd16384 + 17'(eta_q));
    prod    = g * gain;
    rounded = {1'b0, prod} + 33'd8192;
    ghat    = rounded[14 +: GHAT_BITS];
  end
endmodule
