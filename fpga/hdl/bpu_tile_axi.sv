// bpu_tile_axi: tile_core behind AXI4-Lite control registers, one 64-bit AXI4-Stream slave (loads) and one
// 64-bit AXI4-Stream master (results).  Register map (byte offsets, 32-bit):
//   0x00 CTRL     W  bit0 start | bit1 clear_clipped | bit2 emit results | bit3 restart load pointer
//   0x04 STATUS   R  bit0 busy | bit1 done (sticky, cleared by start) | bit2 result stream active
//   0x08 CONFIG   RW [4:0] adc_shift | [5] two_pass | [6] mode_prng | [9:8] result_mode
//   0x0C LOAD_SEL RW [2:0] stream target: 0 weights 1 mask 2 act 3 noise 4 prng state (write also restarts the pointer)
//   0x10 SIGMA_Q  RW [15:0]
//   0x14 LD_COUNT R  beats accepted since the last restart
//   0x18 CLIPPED  R  columns that hit an ADC rail (sticky counter)
//   0x1C READS    R  reads completed
//   0x20 VERSION  R  0xB9000001
module bpu_tile_axi
  import bpu_pkg::*;
#(
  parameter ZLUT_FILE = "zlut_q13.mem"
)(
  input  logic        aclk,
  input  logic        aresetn,
  // AXI4-Lite slave
  input  logic [7:0]  s_axil_awaddr,  input  logic s_axil_awvalid, output logic s_axil_awready,
  input  logic [31:0] s_axil_wdata,   input  logic [3:0] s_axil_wstrb, input logic s_axil_wvalid, output logic s_axil_wready,
  output logic [1:0]  s_axil_bresp,   output logic s_axil_bvalid,  input  logic s_axil_bready,
  input  logic [7:0]  s_axil_araddr,  input  logic s_axil_arvalid, output logic s_axil_arready,
  output logic [31:0] s_axil_rdata,   output logic [1:0] s_axil_rresp, output logic s_axil_rvalid, input logic s_axil_rready,
  // AXI4-Stream slave (loads)
  input  logic [63:0] s_axis_tdata,   input  logic s_axis_tvalid, output logic s_axis_tready, input logic s_axis_tlast,
  // AXI4-Stream master (results)
  output logic [63:0] m_axis_tdata,   output logic m_axis_tvalid, input  logic m_axis_tready, output logic m_axis_tlast
);
  // ---------------- registers
  logic [31:0] cfg_r, load_sel_r, sigma_r;
  logic        start_p, clear_clipped_p, emit_p, restart_p;
  logic        busy, done, done_sticky;
  logic [15:0] ld_count; logic [31:0] clipped_count, read_count;
  logic [8:0]  res_addr; logic [63:0] res_data; logic [9:0] res_len;

  // write channel: accept address+data together
  logic aw_hs;
  assign aw_hs = s_axil_awvalid && s_axil_wvalid && !s_axil_bvalid;
  assign s_axil_awready = aw_hs;
  assign s_axil_wready  = aw_hs;
  assign s_axil_bresp   = 2'b00;
  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      cfg_r <= '0; load_sel_r <= '0; sigma_r <= '0; s_axil_bvalid <= 1'b0;
      start_p <= 1'b0; clear_clipped_p <= 1'b0; emit_p <= 1'b0; restart_p <= 1'b0;
    end else begin
      start_p <= 1'b0; clear_clipped_p <= 1'b0; emit_p <= 1'b0; restart_p <= 1'b0;
      if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;
      if (aw_hs) begin
        s_axil_bvalid <= 1'b1;
        case (s_axil_awaddr[7:2])
          6'h0: begin start_p <= s_axil_wdata[0]; clear_clipped_p <= s_axil_wdata[1]; emit_p <= s_axil_wdata[2]; restart_p <= s_axil_wdata[3]; end
          6'h2: cfg_r      <= s_axil_wdata;
          6'h3: begin load_sel_r <= s_axil_wdata; restart_p <= 1'b1; end
          6'h4: sigma_r    <= s_axil_wdata;
          default: ;
        endcase
      end
    end
  end
  // read channel
  logic emitting;
  assign s_axil_arready = !s_axil_rvalid;
  assign s_axil_rresp   = 2'b00;
  always_ff @(posedge aclk) begin
    if (!aresetn) begin s_axil_rvalid <= 1'b0; s_axil_rdata <= '0; end
    else begin
      if (s_axil_rvalid && s_axil_rready) s_axil_rvalid <= 1'b0;
      if (s_axil_arvalid && s_axil_arready) begin
        s_axil_rvalid <= 1'b1;
        case (s_axil_araddr[7:2])
          6'h1: s_axil_rdata <= {29'b0, emitting, done_sticky, busy};
          6'h2: s_axil_rdata <= cfg_r;
          6'h3: s_axil_rdata <= load_sel_r;
          6'h4: s_axil_rdata <= sigma_r;
          6'h5: s_axil_rdata <= {16'b0, ld_count};
          6'h6: s_axil_rdata <= clipped_count;
          6'h7: s_axil_rdata <= read_count;
          6'h8: s_axil_rdata <= VERSION;
          default: s_axil_rdata <= 32'hDEAD_0000;
        endcase
      end
    end
  end
  always_ff @(posedge aclk) begin
    if (!aresetn) done_sticky <= 1'b0;
    else if (start_p) done_sticky <= 1'b0;
    else if (done) done_sticky <= 1'b1;
  end

  // ---------------- load stream: always ready while idle (the core absorbs one beat per clock)
  assign s_axis_tready = !busy;

  // ---------------- result stream: on emit, walk res_addr 0..res_len-1 (sync read-back, 1 cycle)
  logic [9:0]  emit_idx;
  logic        fetch_valid;
  always_ff @(posedge aclk) begin
    if (!aresetn) begin emitting <= 1'b0; emit_idx <= '0; m_axis_tvalid <= 1'b0; m_axis_tdata <= '0; m_axis_tlast <= 1'b0; fetch_valid <= 1'b0; end
    else begin
      if (emit_p && !emitting) begin emitting <= 1'b1; emit_idx <= '0; fetch_valid <= 1'b0; end
      if (emitting) begin
        if (!m_axis_tvalid || m_axis_tready) begin
          // present the word fetched last cycle, fetch the next
          if (fetch_valid) begin
            m_axis_tvalid <= 1'b1; m_axis_tdata <= res_data; m_axis_tlast <= (emit_idx == res_len);
            if (emit_idx == res_len) begin emitting <= 1'b0; fetch_valid <= 1'b0; end
          end else m_axis_tvalid <= 1'b0;
          if (emit_idx < res_len) begin
            emit_idx <= emit_idx + 10'd1;
            fetch_valid <= 1'b1;
          end else fetch_valid <= 1'b0;
        end
      end else if (m_axis_tvalid && m_axis_tready) m_axis_tvalid <= 1'b0;
    end
  end
  assign res_addr = emit_idx[8:0];

  tile_core #(.ZLUT_FILE(ZLUT_FILE)) u_core (
    .clk(aclk), .rst_n(aresetn),
    .adc_shift(cfg_r[4:0]), .two_pass(cfg_r[5]), .mode_prng(cfg_r[6]), .result_mode(cfg_r[9:8]), .sigma_q(sigma_r[15:0]),
    .ld_valid(s_axis_tvalid && s_axis_tready), .ld_sel(load_sel_r[2:0]), .ld_data(s_axis_tdata), .ld_restart(restart_p), .ld_count(ld_count),
    .start(start_p), .busy(busy), .done(done), .clear_clipped(clear_clipped_p), .clipped_count(clipped_count), .read_count(read_count),
    .res_addr(res_addr), .res_data(res_data), .res_len(res_len));
endmodule
