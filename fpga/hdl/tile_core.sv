// tile_core: the 32x32 differential tile emulator (one tile, contract sections 3-5) behind a plain
// load / start / result interface; bpu_tile_axi.sv wraps it in AXI.
//
// Load port (ld_sel):  0 weights  256 beats x 64 : 4 synapses per beat, {w-[7:0], w+[7:0]} each, row-major
//                      1 mask      16 beats x 64 : bit k = synapse k (row-major); beat b = rows 2b (lo), 2b+1 (hi)
//                      2 act        8 beats x 64 : 4 x int16 per beat
//                      3 noise    512 beats x 64 : 4 x int16 per beat, order ((row*32+col)*2+device)
//                      4 prng     256 beats, low 32 bits : xoshiro state word (instance*4 + j)
// ld_restart resets the beat pointer of the current selection.
//
// Read: start -> [prng fill] -> pass 0 (32 rows) -> [pass 1] -> ADC -> results -> done (1-cycle pulse).
// Result read-back (sync, 1 cycle): result_mode 0: {y_adc[7:0], 16'b0, Y[39:0]}, 32 entries
//                                   result_mode 1: {I-[31:0], I+[31:0]} of the last pass, 32 entries
//                                   result_mode 2: the noise buffer as loaded / generated, 512 entries
module tile_core
  import bpu_pkg::*;
#(
  parameter ZLUT_FILE = "zlut_q13.mem"
)(
  input  logic        clk,
  input  logic        rst_n,
  // configuration
  input  logic [4:0]  adc_shift,
  input  logic        two_pass,
  input  logic        mode_prng,
  input  logic [1:0]  result_mode,
  input  logic [15:0] sigma_q,
  // load
  input  logic        ld_valid,
  input  logic [2:0]  ld_sel,
  input  logic [63:0] ld_data,
  input  logic        ld_restart,
  output logic [15:0] ld_count,
  // control / status
  input  logic        start,
  output logic        busy,
  output logic        done,
  input  logic        clear_clipped,
  output logic [31:0] clipped_count,
  output logic [31:0] read_count,
  // results
  input  logic [8:0]  res_addr,
  output logic [63:0] res_data,
  output logic [9:0]  res_len
);
  // ------------------------------------------------------------------ memories
  logic [63:0] wmem  [8][ROWS];      // bank k: synapses 4k..4k+3 of the row
  logic [31:0] mmem  [ROWS];
  logic [15:0] amem  [ROWS];
  logic [63:0] nmem  [16][ROWS];     // bank k: samples 4k..4k+3 of the row ((col*2+device) indexing)

  // ------------------------------------------------------------------ load
  logic [15:0] ld_ptr;
  logic        prng_ld_en;
  logic [7:0]  prng_ld_idx;
  logic [31:0] prng_ld_word;
  always_ff @(posedge clk) begin
    if (!rst_n || ld_restart) ld_ptr <= '0;
    else if (ld_valid) begin
      ld_ptr <= ld_ptr + 16'd1;
      case (ld_sel)
        3'd0: wmem[ld_ptr[2:0]][ld_ptr[7:3]] <= ld_data;
        3'd1: begin mmem[{ld_ptr[3:0], 1'b0}] <= ld_data[31:0]; mmem[{ld_ptr[3:0], 1'b1}] <= ld_data[63:32]; end
        3'd2: begin
          amem[{ld_ptr[2:0], 2'd0}] <= ld_data[15:0];  amem[{ld_ptr[2:0], 2'd1}] <= ld_data[31:16];
          amem[{ld_ptr[2:0], 2'd2}] <= ld_data[47:32]; amem[{ld_ptr[2:0], 2'd3}] <= ld_data[63:48];
        end
        3'd3: nmem[ld_ptr[3:0]][ld_ptr[8:4]] <= ld_data;
        default: ;
      endcase
    end
  end
  assign ld_count     = ld_ptr;
  assign prng_ld_en   = ld_valid && (ld_sel == 3'd4);
  assign prng_ld_idx  = ld_ptr[7:0];
  assign prng_ld_word = ld_data[31:0];

  // ------------------------------------------------------------------ PRNG
  logic                 prng_step;
  logic [N_GEN*16-1:0]  prng_eta;
  eta_prng #(.ZLUT_FILE(ZLUT_FILE)) u_prng (
    .clk(clk), .ld_en(prng_ld_en), .ld_idx(prng_ld_idx), .ld_word(prng_ld_word),
    .step(prng_step), .sigma_q(sigma_q), .eta(prng_eta));

  // ------------------------------------------------------------------ control FSM
  typedef enum logic [3:0] {S_IDLE, S_FILL, S_ROWS, S_DRAIN, S_FOLD, S_ADC, S_STORE, S_DONE} state_t;
  state_t st;
  logic [5:0] row;            // 0..32
  logic       pass;
  logic [2:0] drain;
  logic [5:0] fill_row;
  logic       fill_wr, fill_wr_d;
  logic [5:0] fill_wr_row, fill_wr_row_d;
  logic [4:0] store_idx;
  logic       adc_start;
  logic       adc_done;

  // pipeline valid: issue -> (1) mem regs -> (2) ghat regs -> (3) term regs -> integrate
  logic [3:0] vpipe;
  logic       issue;
  assign issue = (st == S_ROWS);

  // stage 1 registers (memory read)
  logic [511:0] wrow_r;
  logic [31:0]  mrow_r;
  logic [1023:0] nrow_r;
  logic [15:0]  x_r;
  logic         pass_r;
  always_ff @(posedge clk) begin
    for (int k = 0; k < 8; k++)  wrow_r[k*64 +: 64] <= wmem[k][row[4:0]];
    for (int k = 0; k < 16; k++) nrow_r[k*64 +: 64] <= nmem[k][row[4:0]];
    mrow_r <= mmem[row[4:0]];
    x_r    <= amem[row[4:0]];
    pass_r <= pass;
  end

  // unpack the row for the crossbar
  logic [COLS*W_BITS-1:0] w_p, w_m;
  logic [COLS*16-1:0]     eta_p, eta_m;
  always_comb begin
    for (int c = 0; c < COLS; c++) begin
      w_p[c*8 +: 8]    = wrow_r[c*16 +: 8];
      w_m[c*8 +: 8]    = wrow_r[c*16 + 8 +: 8];
      eta_p[c*16 +: 16] = nrow_r[(c*2)*16 +: 16];
      eta_m[c*16 +: 16] = nrow_r[(c*2+1)*16 +: 16];
    end
  end

  logic signed [DRIVE_BITS-1:0] d;
  wl_dac u_dac (.x_code(x_r), .two_pass(two_pass), .pass_idx(pass_r), .d(d));

  logic [COLS*TERM_BITS-1:0] term_p, term_m;
  xbar_tile u_xbar (.clk(clk), .d(d), .w_p(w_p), .w_m(w_m), .mask(mrow_r), .eta_p(eta_p), .eta_m(eta_m),
                    .term_p(term_p), .term_m(term_m));

  logic [COLS*CUR_BITS-1:0] i_p, i_m;
  logic int_clear;
  col_integrate u_int (.clk(clk), .clear(int_clear), .en(vpipe[3]), .term_p(term_p), .term_m(term_m), .i_p(i_p), .i_m(i_m));

  logic [COLS*ACC_BITS-1:0] y_acc;
  logic acc_clear, fold_en, fold_pass;
  shift_add u_sa (.clk(clk), .clear(acc_clear), .en(fold_en), .pass_idx(fold_pass), .i_p(i_p), .i_m(i_m), .y_acc(y_acc));

  logic [COLS*ADC_BITS-1:0] codes;
  logic [COLS-1:0] clips, adc_dones;
  genvar c;
  generate
    for (c = 0; c < COLS; c++) begin : g_adc
      sar_adc u_adc (.clk(clk), .start(adc_start), .shift(adc_shift), .y($signed(y_acc[c*ACC_BITS +: ACC_BITS])),
                     .code(codes[c*8 +: 8]), .clipped(clips[c]), .done(adc_dones[c]));
    end
  endgenerate
  assign adc_done = adc_dones[0];

  // result buffer
  logic [63:0] rmem [ROWS];
  logic [63:0] res_q;
  logic [63:0] noise_q;
  always_ff @(posedge clk) begin
    res_q   <= rmem[res_addr[4:0]];
    noise_q <= nmem[res_addr[3:0]][res_addr[8:4]];
  end
  assign res_data = (result_mode == 2'd2) ? noise_q : res_q;
  assign res_len  = (result_mode == 2'd2) ? 10'd512 : 10'd32;

  // prng fill writes
  always_ff @(posedge clk) begin
    if (fill_wr)
      for (int k = 0; k < 16; k++) nmem[k][fill_wr_row[4:0]] <= prng_eta[k*64 +: 64];
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st <= S_IDLE; busy <= 1'b0; done <= 1'b0; vpipe <= '0; row <= '0; pass <= 1'b0;
      int_clear <= 1'b0; acc_clear <= 1'b0; fold_en <= 1'b0; adc_start <= 1'b0; prng_step <= 1'b0;
      fill_wr <= 1'b0; fill_wr_d <= 1'b0; clipped_count <= '0; read_count <= '0; store_idx <= '0; drain <= '0; fill_row <= '0;
    end else begin
      done <= 1'b0; int_clear <= 1'b0; acc_clear <= 1'b0; fold_en <= 1'b0; adc_start <= 1'b0;
      prng_step <= 1'b0; fill_wr <= 1'b0; fill_wr_d <= 1'b0;
      vpipe <= {vpipe[2:0], issue};
      if (clear_clipped) clipped_count <= '0;
      case (st)
        S_IDLE: if (start) begin
          busy <= 1'b1; pass <= 1'b0; row <= '0; acc_clear <= 1'b1; int_clear <= 1'b1;
          fill_row <= '0;
          if (mode_prng) st <= S_FILL; else st <= S_ROWS;
        end
        S_FILL: begin                                   // 32 steps; eta is valid two cycles after each step
          if (fill_row < 6'd32) begin prng_step <= 1'b1; fill_row <= fill_row + 6'd1; end
          fill_wr_d     <= prng_step;
          fill_wr_row_d <= fill_row - 6'd1;
          fill_wr       <= fill_wr_d;
          fill_wr_row   <= fill_wr_row_d;
          if (fill_row == 6'd32 && !prng_step && !fill_wr_d) st <= S_ROWS;
        end
        S_ROWS: begin
          row <= row + 6'd1;
          if (row == 6'd31) begin st <= S_DRAIN; drain <= 3'd0; end
        end
        S_DRAIN: begin                                  // let the last row reach the integrators
          drain <= drain + 3'd1;
          if (drain == 3'd3) st <= S_FOLD;
        end
        S_FOLD: begin
          fold_en <= 1'b1; fold_pass <= pass;      // the fold of pass p happens one cycle later, when `pass` may already be p+1
          if (two_pass && !pass) begin
            pass <= 1'b1; row <= '0; int_clear <= 1'b1; st <= S_ROWS;
          end else begin
            st <= S_ADC;
            drain <= 3'd0;
          end
        end
        S_ADC: begin
          if (drain == 3'd0) begin adc_start <= 1'b1; drain <= 3'd1; end
          else if (adc_done) begin st <= S_STORE; store_idx <= '0; end
        end
        S_STORE: begin
          rmem[store_idx] <= (result_mode == 2'd1)
            ? {i_m[store_idx*CUR_BITS +: CUR_BITS], i_p[store_idx*CUR_BITS +: CUR_BITS]}
            : {codes[store_idx*8 +: 8], 16'b0, y_acc[store_idx*ACC_BITS +: ACC_BITS]};
          store_idx <= store_idx + 5'd1;
          if (store_idx == 5'd31) st <= S_DONE;
        end
        S_DONE: begin
          clipped_count <= clipped_count + 32'($countones(clips));
          read_count <= read_count + 32'd1;
          busy <= 1'b0; done <= 1'b1; st <= S_IDLE;
        end
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
