# Code Block Test Document

## 1. Normal code block

Here is a properly formatted code block:

```verilog
module test;
    initial begin
        $display("Hello World");
    end
endmodule
```

## 2. Prose trapped in code block

```
module counter (
    input  logic clk,
    output logic [7:0] count
);

## 3. This heading is inside the code block by mistake

This paragraph is also incorrectly inside the code block.
It has multiple sentences and is clearly not code.

    always_ff @(posedge clk)
        count <= count + 1;
endmodule
```

## 4. Indented fence problem

The following code block has an indented fence:

```vhdl
entity adder is
    port (
        a, b : in  std_logic_vector(7 downto 0);
        sum  : out std_logic_vector(7 downto 0)
    );
end entity adder;

architecture rtl of adder is
begin
    sum <= std_logic_vector(unsigned(a) + unsigned(b));
end architecture rtl;
    ```

Some text after the indented fence.

## 5. Unclosed code block

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

## 6. Another section after unclosed block

This section should not be inside any code block.

## 7. Adjacent code blocks

```verilog
always_ff @(posedge clk) begin
    if (reset)
        data <= 0;
```

```
    else
        data <= next_data;
end
```

## 8. Clean section

This section has no code block issues.

It is just regular markdown text with a list:

- Item one
- Item two
- Item three
