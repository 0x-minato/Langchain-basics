from fastmcp import FastMCP
import math

mcp = FastMCP('Math')

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to exponent."""
    return base ** exponent

@mcp.tool()
def square_root(n: float) -> float:
    """Square root of n."""
    if n < 0:
        raise ValueError("Cannot take square root of a negative number")
    return math.sqrt(n)

@mcp.tool()
def percentage(part: float, whole: float) -> float:
    """What percent is part of whole."""
    if whole == 0:
        raise ValueError("Whole cannot be zero")
    return (part / whole) * 100

if __name__ == "__main__":
    mcp.run()
