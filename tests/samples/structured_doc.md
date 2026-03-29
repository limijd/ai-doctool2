# JTAG Boundary Scan Architecture

## 1. Introduction

This document describes the JTAG boundary-scan architecture
defined by IEEE Std 1149.1.

## 1.1 Purpose

The boundary-scan architecture provides a standardized method
for testing interconnections between integrated circuits.

## 1.2 Scope

This specification covers the test access port (TAP) and
boundary-scan register (BSR) design.

## 2. TAP Controller

The TAP controller is a synchronous finite state machine.

### 2.1 State diagram

The controller has 16 states organized in two main paths:
the Data Register (DR) path and the Instruction Register (IR) path.

### 2.2 Signals

| Signal | Direction | Description |
|--------|-----------|-------------|
| TCK    | Input     | Test Clock  |
| TMS    | Input     | Test Mode Select |
| TDI    | Input     | Test Data In |
| TDO    | Output    | Test Data Out |
| TRST*  | Input     | Test Reset (optional) |

## 3. Registers

### 3.1 Boundary scan register

The BSR consists of boundary-scan cells connected between
each device pin and the core logic.

```verilog
module boundary_cell (
    input  logic data_in,
    input  logic scan_in,
    input  logic shift_dr,
    input  logic clock_dr,
    output logic data_out,
    output logic scan_out
);
    logic capture_ff, update_ff;

    always_ff @(posedge clock_dr)
        capture_ff <= shift_dr ? scan_in : data_in;

    always_ff @(negedge clock_dr)
        update_ff <= capture_ff;

    assign data_out = update_ff;
    assign scan_out = capture_ff;
endmodule
```

### 3.2 Instruction register

The instruction register holds the current test instruction.

Supported instructions:
- **BYPASS**: Shortest path from TDI to TDO
- **EXTEST**: Drive and capture boundary pins
- **SAMPLE**: Sample boundary pins without driving
- **PRELOAD**: Load boundary register before EXTEST

## 4. Implementation notes

The boundary-scan architecture adds minimal area overhead,
typically less than 5% of the total gate count.

### 4.1 Timing considerations

All scan operations are synchronous to TCK.
The maximum TCK frequency depends on the implementation.

### 4.2 Power considerations

During normal operation, the TAP controller remains in
the Test-Logic-Reset state, consuming minimal power.
