// BPU FPGA tile: shared constants.  Every value here is fixed by docs/reference_contract.md
// and bpusim/fixed.py; change them there first.
package bpu_pkg;
  localparam int ROWS       = 32;
  localparam int COLS       = 32;
  localparam int W_BITS     = 8;
  localparam int GMIN_Q     = 655;
  localparam int GSTEP      = 254;
  localparam int ETA_FRAC   = 14;
  localparam int ETA_ONE    = 1 << ETA_FRAC;
  localparam int ETA_MAX    = ETA_ONE - 1;
  localparam int GHAT_BITS  = 17;
  localparam int DRIVE_BITS = 9;     // DAC row drive, signed
  localparam int TERM_BITS  = 27;    // 9 x 18 signed product
  localparam int CUR_BITS   = 32;    // I+ / I- per pass
  localparam int ACC_BITS   = 40;    // Y
  localparam int ADC_BITS   = 8;
  localparam int ZLUT_BITS  = 10;
  localparam int ZLUT_FRAC  = 13;
  localparam int N_GEN      = 2 * COLS;   // one xoshiro per (column, device)
  localparam logic [31:0] VERSION = 32'hB9000001;
endpackage
