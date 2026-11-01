// sar_adc: 8-bit successive-approximation converter model, bit-exact to the contract (section 3 step 8).
//   q = rne_shift(Y, S)  (round half to even on the arithmetic shift), then an 8-cycle SAR search of
//   the offset-binary value u = q + 128 against trial codes; the comparator saturates naturally:
//   u >= 255 -> 255, u < 0 -> 0.  code = acc - 128.  clipped = (q < -128) | (q > 127).
module sar_adc
  import bpu_pkg::*;
(
  input  logic                        clk,
  input  logic                        start,
  input  logic [4:0]                  shift,
  input  logic signed [ACC_BITS-1:0]  y,
  output logic signed [ADC_BITS-1:0]  code,
  output logic                        clipped,
  output logic                        done
);
  // ---- rounding (combinational on the registered input) ----
  logic signed [ACC_BITS-1:0] r;            // floor(y / 2^S)
  logic [ACC_BITS-1:0]        mask, rem, half;
  logic                       up;
  logic signed [ACC_BITS:0]   q;            // one bit wider for +1 and +128
  logic signed [ACC_BITS:0]   u;
  always_comb begin
    r    = y >>> shift;
    mask = (40'd1 << shift) - 40'd1;
    rem  = y & mask;
    half = (shift == 5'd0) ? 40'd0 : (40'd1 << (shift - 5'd1));
    up   = (shift != 5'd0) && ((rem > half) || ((rem == half) && r[0]));
    q    = (ACC_BITS+1)'(r) + (up ? 41'sd1 : 41'sd0);
    u    = q + 41'sd128;
  end
  // ---- SAR search ----
  logic [ADC_BITS-1:0] acc;
  logic [3:0]          bitpos;             // 8 .. 0
  logic                busy;
  logic signed [ACC_BITS:0] u_held;
  logic [ADC_BITS-1:0] trial;
  always_comb trial = acc | (8'd1 << (bitpos - 4'd1));
  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (start) begin
      acc    <= '0;
      bitpos <= 4'd8;
      busy   <= 1'b1;
      u_held <= u;
      clipped <= (q < -41'sd128) || (q > 41'sd127);
    end else if (busy) begin
      if ($signed({33'b0, trial}) <= u_held) acc <= trial;
      if (bitpos == 4'd1) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
      bitpos <= bitpos - 4'd1;
    end
  end
  assign code = $signed(acc) - 8'sd128;   // offset binary -> two's complement (mod 2^8)
endmodule
