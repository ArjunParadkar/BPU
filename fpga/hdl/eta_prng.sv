// eta_prng: mode B noise source (contract section 5B).  N_GEN = 64 xoshiro128** generators, instance
// m = 2*col + device.  Each 'step' advances all 64 once and presents 64 eta_q samples:
//   k = word >> 22 ; z = ZLUT[k] (Q3.13) ; eta = clamp((sigma_q * z + 4096) >>> 13, +-16383)
// State is loaded by the PS (splitmix64 is done on the PS side, see the contract), 128 bits per instance.
// The Gaussian ROM is read synchronously (so Vivado can put the 64 copies in BRAM18s): eta is valid TWO clocks
// after step.  Constant multiplies are written as shift-adds so no DSP48E2 is spent on them.
module eta_prng
  import bpu_pkg::*;
#(
  parameter ZLUT_FILE = "zlut_q13.mem"
)(
  input  logic                clk,
  input  logic                ld_en,          // load one 32-bit state word
  input  logic [7:0]          ld_idx,         // 0 .. 255 : instance = idx >> 2, word = idx & 3
  input  logic [31:0]         ld_word,
  input  logic                step,
  input  logic [15:0]         sigma_q,
  output logic [N_GEN*16-1:0] eta             // 64 x int16, valid two cycles after step
);
  logic [15:0] zlut [0:(1<<ZLUT_BITS)-1];
  initial $readmemh(ZLUT_FILE, zlut);

  logic [31:0] s0 [N_GEN], s1 [N_GEN], s2 [N_GEN], s3 [N_GEN];

  function automatic logic [31:0] rotl32(input logic [31:0] x, input int k);
    rotl32 = (x << k) | (x >> (32 - k));
  endfunction

  genvar m;
  generate
    for (m = 0; m < N_GEN; m++) begin : g_gen
      logic [31:0] x5, result, t;
      logic [31:0] n0, n1, n2, n3;
      logic signed [15:0] z;          // ROM output, registered
      logic               z_valid;
      logic signed [32:0] prod;
      logic signed [19:0] e;
      logic signed [15:0] e_clamped;
      always_comb begin
        x5     = (s1[m] << 2) + s1[m];                       // * 5
        result = (rotl32(x5, 7) << 3) + rotl32(x5, 7);       // * 9
        t  = s1[m] << 9;
        n2 = s2[m] ^ s0[m];
        n3 = s3[m] ^ s1[m];
        n1 = s1[m] ^ n2;
        n0 = s0[m] ^ n3;
        n2 = n2 ^ t;
        n3 = rotl32(n3, 11);
        prod = $signed({1'b0, sigma_q}) * z;
        e    = 20'(($signed(prod) + 33'sd4096) >>> ZLUT_FRAC);
        if (e > 20'sd16383)        e_clamped = 16'sd16383;
        else if (e < -20'sd16383)  e_clamped = -16'sd16383;
        else                       e_clamped = 16'(e);
      end
      always_ff @(posedge clk) begin
        z_valid <= step;
        if (step) z <= $signed(zlut[result[31 -: ZLUT_BITS]]);   // synchronous ROM read (BRAM)
        if (z_valid) eta[m*16 +: 16] <= e_clamped;
        if (ld_en && (ld_idx[7:2] == m)) begin
          case (ld_idx[1:0])
            2'd0: s0[m] <= ld_word;
            2'd1: s1[m] <= ld_word;
            2'd2: s2[m] <= ld_word;
            default: s3[m] <= ld_word;
          endcase
        end else if (step) begin
          s0[m] <= n0; s1[m] <= n1; s2[m] <= n2; s3[m] <= n3;
        end
      end
    end
  endgenerate
endmodule
