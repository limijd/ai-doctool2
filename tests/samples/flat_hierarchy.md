## IEEE Standard for SystemVerilog

## 1. Overview

This standard defines the SystemVerilog language.

## 1.1 Scope

The scope of this standard covers the SystemVerilog hardware description
and verification language.

## 1.1.1 Language features

SystemVerilog extends Verilog with advanced features for verification.

## 1.2 Purpose

The purpose is to provide a unified language for design and verification.

## 2. Normative references

The following documents are referenced:

## 2.1 General

References are listed in this section.

## 3. Design and verification constructs

This chapter covers the core language constructs.

## 3.1 Modules

## Rules

Module declarations follow specific rules.

## Permissions

Implementations may extend the syntax.

## 3.1.1 Module declarations

A module is the basic building block.

## 3.1.2 Port declarations

Ports define the interface of a module.

## Example

```systemverilog
module counter (
    input  logic clk,
    input  logic rst,
    output logic [7:0] count
);
    always_ff @(posedge clk or posedge rst) begin
        if (rst) count <= 8'b0;
        else     count <= count + 1;
    end
endmodule
```

## 3.2 Interfaces

## 3.2.1 Interface declarations

Interfaces bundle signals together.

## Notes

Interfaces are synthesizable in most tools.

## A. Formal syntax

This annex defines the formal syntax.

## A.1 Source text

Grammar rules for source text.

## A.1.1 Library source text

Library-level grammar rules.

## B. Keywords

Reserved keywords in the language.

## B.1 SystemVerilog keywords

List of all SystemVerilog reserved keywords.
