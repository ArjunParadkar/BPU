// tb_tile_gate: file-driven bit-exact gate.  Reads the golden set written by vectors/gen_vectors.py
// (+dir=<path>), runs every vector through tile_core in stream mode and compares EVERY output bit
// (y_adc and the 40-bit pre-ADC accumulator) with the expectation.  Prints the first divergences as
//   MISMATCH vec=<i> col=<c> got=<hex> exp=<hex>
// and ends with GATE PASS / GATE FAIL.  Also reports cycles per read for docs/RESULTS.md.
`timescale 1ns/1ps
module tb_tile_gate;
  parameter int N_MAX = 10300;
  parameter int W_MAX = 160;
  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  logic [4:0]  adc_shift; logic two_pass; logic mode_prng = 0; logic [1:0] result_mode = 0; logic [15:0] sigma_q = 0;
  logic ld_valid = 0; logic [2:0] ld_sel = 0; logic [63:0] ld_data = 0; logic ld_restart = 0; logic [15:0] ld_count;
  logic start = 0, busy, done, clear_clipped = 0; logic [31:0] clipped_count, read_count;
  logic [8:0] res_addr = 0; logic [63:0] res_data; logic [9:0] res_len;

  tile_core #(.ZLUT_FILE("zlut_q13.mem")) dut (.*);

  logic [63:0] wset_mem  [0:W_MAX*256-1];
  logic [63:0] mask_mem  [0:W_MAX*16-1];
  logic [63:0] cfg_mem   [0:N_MAX-1];
  logic [63:0] act_mem   [0:N_MAX*8-1];
  logic [63:0] noise_mem [0:N_MAX*512-1];
  logic [63:0] exp_mem   [0:N_MAX*32-1];

  string dir;
  int n_vec, n_mismatch, n_shown, cur_wset, cyc, cyc_total, cyc_min, cyc_max, t0;
  int fd;

  task automatic load(input logic [2:0] sel, input int base, input int n, input int which);
    ld_sel = sel; ld_restart = 1; @(posedge clk); ld_restart = 0;
    for (int b = 0; b < n; b++) begin
      ld_valid = 1;
      case (which)
        0: ld_data = wset_mem[base + b];
        1: ld_data = mask_mem[base + b];
        2: ld_data = act_mem[base + b];
        default: ld_data = noise_mem[base + b];
      endcase
      @(posedge clk);
    end
    ld_valid = 0;
  endtask

  initial begin
    if (!$value$plusargs("dir=%s", dir)) dir = "vec";
    if (!$value$plusargs("n=%d", n_vec)) n_vec = 0;
    $readmemh({dir, "/wset.hex"}, wset_mem);
    $readmemh({dir, "/mask.hex"}, mask_mem);
    $readmemh({dir, "/cfg.hex"}, cfg_mem);
    $readmemh({dir, "/act.hex"}, act_mem);
    $readmemh({dir, "/noise.hex"}, noise_mem);
    $readmemh({dir, "/expect.hex"}, exp_mem);
    n_mismatch = 0; n_shown = 0; cur_wset = -1; cyc_total = 0; cyc_min = 1 << 30; cyc_max = 0;
    repeat (3) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
    for (int i = 0; i < n_vec; i++) begin
      int ws; ws = int'(cfg_mem[i][15:0]);
      adc_shift = cfg_mem[i][20:16]; two_pass = cfg_mem[i][21];
      if (ws != cur_wset) begin
        load(3'd0, ws * 256, 256, 0);
        load(3'd1, ws * 16, 16, 1);
        cur_wset = ws;
      end
      load(3'd2, i * 8, 8, 2);
      load(3'd3, i * 512, 512, 3);
      start = 1; @(posedge clk); start = 0; cyc = 0;
      while (!done) begin @(posedge clk); cyc++; end
      cyc_total += cyc; if (cyc < cyc_min) cyc_min = cyc; if (cyc > cyc_max) cyc_max = cyc;
      for (int c = 0; c < 32; c++) begin
        res_addr = c[8:0]; @(posedge clk); #1;
        if (res_data !== exp_mem[i * 32 + c]) begin
          n_mismatch++;
          if (n_shown < 20) begin
            $display("MISMATCH vec=%0d col=%0d got=%016h exp=%016h (S=%0d two_pass=%0d)", i, c, res_data, exp_mem[i*32+c], adc_shift, two_pass);
            n_shown++;
          end
        end
      end
    end
    $display("CYCLES per read: min=%0d max=%0d mean=%0d (load beats not counted)", cyc_min, cyc_max, n_vec ? cyc_total / n_vec : 0);
    $display("CLIPPED columns counted by the DUT: %0d", clipped_count);
    if (n_mismatch == 0) $display("GATE PASS vectors=%0d columns=%0d mismatches=0", n_vec, n_vec * 32);
    else                 $display("GATE FAIL vectors=%0d mismatches=%0d", n_vec, n_mismatch);
    $finish;
  end
endmodule
